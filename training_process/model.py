"""
Reuse-aware dynamic GNN models: WDGCN, TGCN, EvolveGCN, GATLSTM.

All four models exploit three forms of data reuse over consecutive snapshots:

  Type-I (snapshot-level / intra-group reuse)
      Within one snapshot group, consecutive snapshots differ only by a small
      set of edge additions/deletions encoded in a DiffMap (``dmap``). The
      aggregation stage is updated incrementally via IncreConv, which
      processes only the changed edges and combines the delta with the cached
      aggregation from the previous snapshot. For GCN-based models the
      transformation stage (MatMul) is also reused: only nodes whose
      neighborhood changed are fed through the linear projection, the rest
      reuse cached results. For GAT, MatMul is computed once on the first
      snapshot and shared across the group because features are
      time-invariant at the first layer.

  Type-II (group-level / inter-group reuse)
      When two consecutive snapshot groups share overlapping snapshots and
      are co-located on the same GPU, their intermediate GNN outputs can
      persist on GPU and be consumed by the second group, avoiding redundant
      HtoD + GraphConv. The models expose a save/apply pattern.

  Triple-stream pipeline
      Three CUDA streams coordinated by cudaEvents:
          HtoD stream   -- host-to-device graph transfer
          Conv stream   -- sparse aggregation (GraphConv / IncreConv / GAT)
          MatMul stream -- dense linear projection
      For GCN-based models the dependency chain is HtoD -> Conv -> MatMul.
      For GAT the transformation precedes aggregation, so the chain becomes
      HtoD -> MatMul -> Conv; since MatMul is cached, the pipeline
      effectively overlaps HtoD and Conv.

EvolveGCN is a special case: its linear weight evolves per snapshot through a
matrix GRU cell, so MatMul reuse does not apply. Only Type-I aggregation
reuse and HtoD/Conv overlap are used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from dgl.nn.pytorch import GraphConv, GATConv


# =============================================================================
# Intra-group reuse-aware convolution wrappers.
# =============================================================================

class IntraReuseConv(nn.Module):
    """
    GCN wrapper that exposes three primitives for Type-I reuse:
        forward(g, feat)        -- full aggregation on the base snapshot
        incre_aggregate(...)    -- IncreConv on a DiffMap
        unique_gemm(feat)       -- shared linear projection
    """

    def __init__(self, in_feats, hidden_size):
        super().__init__()
        self.gcn = GraphConv(
            in_feats, hidden_size,
            activation=F.relu,
            norm='none', bias=False, weight=False,
            reuse=True,
            allow_zero_in_degree=True,
        )

    def forward(self, g, feat):
        return self.gcn(g, feat)

    def incre_aggregate(self, feat, prev_agg, dmap):
        return self.gcn.increConv_subgraph(feat, prev_agg, dmap)

    def unique_gemm(self, feat):
        return self.gcn.unique_gemm(feat)


class IntraReuseGAT(nn.Module):
    """
    GAT wrapper for Type-I reuse:
        forward(g, feat, reuse=True) -> (out, reuse_cache)
            Runs attention + aggregation on the base snapshot and returns a
            cache (attention scores, softmax denominator, etc.) used by
            increGAT_forward.
        incre_forward(g, reuse_cache, prev_out, changed_nodes)
            Incremental GAT that updates only the softmax denominator and the
            differential aggregation for changed edges.
    """

    def __init__(self, in_feats, hidden_size, num_heads):
        super().__init__()
        self.gat = GATConv(
            in_feats, hidden_size,
            num_heads=num_heads,
            activation=F.relu,
            bias=False,
            residual=False,
            allow_zero_in_degree=True,
        )

    def forward(self, g, feat):
        return self.gat(g, feat, reuse=True)

    def incre_forward(self, g, reuse_cache, prev_out, changed_nodes):
        return self.gat.increGAT_forward(g, reuse_cache, prev_out, changed_nodes)


# =============================================================================
# Intra-group helpers shared by the GCN-based models.
# =============================================================================

def _aggregate_group_gcn(rgcn, g_list, feat):
    """Run the GCN aggregation stage with Type-I reuse on a snapshot group."""
    aggregations = []
    for i, g in enumerate(g_list):
        if i == 0:
            aggregations.append(rgcn(g, feat))
        else:
            aggregations.append(rgcn.incre_aggregate(feat, aggregations[-1], g))
    return aggregations


def _project_with_unique_gemm(rgcn, aggregations, changed_nodes):
    """
    MatMul reuse stage: project the base snapshot in full, but project only
    the changed-node rows for subsequent snapshots. Reconstruct the full
    per-snapshot outputs by cloning and scattering.
    """
    shape_list = [aggregations[0].shape[0]]
    unique_feat = aggregations[0]
    for i in range(1, len(aggregations)):
        changed = changed_nodes[i - 1]
        shape_list.append(len(changed))
        unique_feat = torch.cat((unique_feat, aggregations[i][changed]), dim=0)
    unique_out = rgcn.unique_gemm(unique_feat)

    outs = [unique_out[:shape_list[0]]]
    end = shape_list[0]
    for i in range(1, len(aggregations)):
        full = outs[-1].clone()
        full[changed_nodes[i - 1]] = unique_out[end:end + shape_list[i]]
        end += shape_list[i]
        outs.append(full)
    return outs


# =============================================================================
# Triple-stream pipelines.
# =============================================================================

def triple_stream_pipeline_gcn(rgcn, g_list, changed_nodes, feat, rank):
    """
    Triple-stream pipeline for GCN-based models.

    Streams:
        copy_stream  -- PCIe bandwidth (HtoD)
        conv_stream  -- HBM bandwidth (GraphConv / IncreConv)
        gemm_stream  -- Compute (dense MatMul)
    Dependency: HtoD(i) -> Conv(i) -> MatMul(i).
    HtoD(i+1) and Conv(i) overlap; Conv(i) and MatMul(i-1) overlap.
    """
    copy_stream = torch.cuda.Stream(priority=0)
    conv_stream = torch.cuda.Stream(priority=0)
    gemm_stream = torch.cuda.Stream(priority=-1)

    with torch.cuda.stream(copy_stream):
        g_dev = g_list[0].pin_memory_().to(rank, non_blocking=True)
        copy_done = [torch.cuda.Event()]
        copy_done[0].record(copy_stream)

    conv_done = []
    conv_out_buf = []
    gemm_inputs = []
    gemm_out_buf = []
    g_next_dev = None

    for i in range(len(g_list)):
        # (i+1)'th HtoD overlaps with Conv(i).
        if i + 1 < len(g_list):
            with torch.cuda.stream(copy_stream):
                g_next_dev = g_list[i + 1].pin_memory_().to(rank, non_blocking=True)
                ev_cp = torch.cuda.Event()
                ev_cp.record(copy_stream)
                copy_done.append(ev_cp)

        conv_stream.wait_event(copy_done[i])
        with torch.cuda.stream(conv_stream):
            if i == 0:
                conv_out = rgcn(g_dev, feat)
                gemm_inputs.append(conv_out)
            else:
                conv_out = rgcn.incre_aggregate(feat, conv_out_buf[-1], g_dev)
                gemm_inputs.append(conv_out[changed_nodes[i - 1]])
            conv_out_buf.append(conv_out)
            ev_conv = torch.cuda.Event()
            ev_conv.record(conv_stream)
            conv_done.append(ev_conv)

        if i > 0:
            gemm_stream.wait_event(conv_done[i - 1])
            with torch.cuda.stream(gemm_stream):
                gemm_out_buf.append(rgcn.unique_gemm(gemm_inputs[i - 1]))

        if i + 1 < len(g_list):
            g_dev = g_next_dev

    gemm_stream.wait_event(conv_done[-1])
    with torch.cuda.stream(gemm_stream):
        gemm_out_buf.append(rgcn.unique_gemm(gemm_inputs[-1]))
    torch.cuda.synchronize()

    outs = [gemm_out_buf[0]]
    for i in range(1, len(g_list)):
        out = outs[-1].clone()
        out[changed_nodes[i - 1]] = gemm_out_buf[i]
        outs.append(out)
    return outs


def triple_stream_pipeline_gat(rgat, g_list, changed_nodes, feat, rank):
    """
    Triple-stream pipeline for GAT.

    In GAT the transformation (Wh_i) precedes aggregation, and within a group
    node features are time-invariant, so the transformation is computed once
    and cached. The pipeline therefore overlaps HtoD with the (incremental)
    Conv stream only; MatMul runs implicitly as part of the first GAT call.
    Dependency: HtoD(i) -> Conv(i); Conv(i) consumes the cached MatMul.
    """
    copy_stream = torch.cuda.Stream(priority=0)
    conv_stream = torch.cuda.Stream(priority=0)

    with torch.cuda.stream(copy_stream):
        g_dev = g_list[0].pin_memory_().to(rank, non_blocking=True)
        copy_done = [torch.cuda.Event()]
        copy_done[0].record(copy_stream)

    conv_out_buf = []
    reuse_cache = None
    g_next_dev = None

    for i in range(len(g_list)):
        if i + 1 < len(g_list):
            with torch.cuda.stream(copy_stream):
                g_next_dev = g_list[i + 1].pin_memory_().to(rank, non_blocking=True)
                ev_cp = torch.cuda.Event()
                ev_cp.record(copy_stream)
                copy_done.append(ev_cp)

        conv_stream.wait_event(copy_done[i])
        with torch.cuda.stream(conv_stream):
            if i == 0:
                out, reuse_cache = rgat(g_dev, feat)
            else:
                out = rgat.incre_forward(
                    g_dev, reuse_cache, conv_out_buf[-1], changed_nodes[i - 1])
            conv_out_buf.append(out)

        if i + 1 < len(g_list):
            g_dev = g_next_dev

    torch.cuda.synchronize()
    return conv_out_buf


# =============================================================================
# Temporal heads: mini-batched LSTM / GRU to bound peak memory on large N.
# =============================================================================

def _rnn_last_step(rnn, per_snapshot_feats, hidden_state=None,
                   batch_nodes=4096 * 256):
    """Stack features into [N, T, F], run the RNN in node-batches, return
    the last timestep's hidden for each node."""
    seq = torch.stack(per_snapshot_feats, dim=1)
    out_chunks = []
    final_hidden = None
    for start in range(0, seq.size(0), batch_nodes):
        batch = seq[start:start + batch_nodes]
        if hidden_state is None:
            out, final_hidden = rnn(batch)
        else:
            out, final_hidden = rnn(batch, hidden_state)
        out_chunks.append(out)
    full = torch.cat(out_chunks, dim=0)
    return full[:, -1, :], final_hidden


# =============================================================================
# Mixin that provides the four reuse entry points for GCN-based RNN models.
# =============================================================================

class _GCNReuseMixin:
    """
    Subclasses must provide:
        self.rgcn        : IntraReuseConv
        self._head(x)    : takes [N, hidden] -> classifier logits
        self._rnn        : nn.LSTM / nn.GRU (used by self._temporal)
    """

    def _temporal(self, per_snapshot_feats):
        last, _ = _rnn_last_step(self._rnn, per_snapshot_feats)
        return self._head(last)

    def forward_intra(self, g_list, changed_nodes, feat):
        aggregations = _aggregate_group_gcn(self.rgcn, g_list, feat)
        outs = _project_with_unique_gemm(self.rgcn, aggregations, changed_nodes)
        return self._temporal(outs)

    def forward_intra_3stream(self, g_list, changed_nodes, feat, rank):
        outs = triple_stream_pipeline_gcn(
            self.rgcn, g_list, changed_nodes, feat, rank)
        return self._temporal(outs)

    def forward_inter_save(self, g_list, feat, changed_nodes, inter_reuse_idx):
        aggregations = _aggregate_group_gcn(self.rgcn, g_list, feat)
        cached = [aggregations[i] for i in inter_reuse_idx]
        outs = _project_with_unique_gemm(self.rgcn, aggregations, changed_nodes)
        return self._temporal(outs), cached

    def forward_inter_apply(self, g_list, feat, changed_nodes,
                            inter_reuse_idx, cached_aggregations):
        if not inter_reuse_idx:
            return self.forward_intra(g_list, changed_nodes, feat)
        # Aggregations: pull cached entries at overlap positions; compute the
        # rest via IncreConv on the non-overlapping snapshots.
        aggregations = [None] * len(g_list)
        cached_iter = iter(cached_aggregations)
        for i in range(len(g_list)):
            if i in inter_reuse_idx:
                aggregations[i] = next(cached_iter)
                continue
            if i == 0:
                aggregations[i] = self.rgcn(g_list[i], feat)
            else:
                # Fall back to the previous aggregation (whether cached or
                # freshly computed) as the base for IncreConv.
                aggregations[i] = self.rgcn.incre_aggregate(
                    feat, aggregations[i - 1], g_list[i])
        outs = _project_with_unique_gemm(self.rgcn, aggregations, changed_nodes)
        return self._temporal(outs)


# =============================================================================
# WDGCN: GCN + LSTM.
# =============================================================================

class WDGCN(nn.Module, _GCNReuseMixin):
    def __init__(self, in_feats, hidden_size, out_feats):
        super().__init__()
        self.rgcn = IntraReuseConv(in_feats, hidden_size)
        self._rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_feats)

    def _head(self, x):
        return self.fc(x)


# =============================================================================
# TGCN: GCN + GRU.
# =============================================================================

class TGCN(nn.Module, _GCNReuseMixin):
    def __init__(self, in_feats, hidden_size, out_feats):
        super().__init__()
        self.rgcn = IntraReuseConv(in_feats, hidden_size)
        self._rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_feats)

    def _head(self, x):
        return self.fc(x)


# =============================================================================
# EvolveGCN: GCN with per-snapshot weight evolution + MLP classifier.
# =============================================================================

class _MatGRUGate(nn.Module):
    """Matrix-valued GRU gate used by EvolveGCN."""

    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows, rows))
        self.U = Parameter(torch.Tensor(rows, rows))
        self.bias = Parameter(torch.Tensor(rows, cols))
        init.xavier_uniform_(self.W)
        init.xavier_uniform_(self.U)
        init.zeros_(self.bias)

    def forward(self, x, hidden):
        return self.activation(self.W @ x + self.U @ hidden + self.bias)


class _MatGRUCell(nn.Module):
    """Matrix-valued GRU cell that updates the GCN weight per snapshot."""

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.update = _MatGRUGate(in_feats, out_feats, nn.Sigmoid())
        self.reset = _MatGRUGate(in_feats, out_feats, nn.Sigmoid())
        self.htilda = _MatGRUGate(in_feats, out_feats, nn.Tanh())

    def forward(self, prev_W, z_topk=None):
        if z_topk is None:
            z_topk = prev_W
        update = self.update(z_topk, prev_W)
        reset = self.reset(z_topk, prev_W)
        h_cap = self.htilda(z_topk, reset * prev_W)
        return (1 - update) * prev_W + update * h_cap


class EvolveGCN(nn.Module):
    """
    EvolveGCN-H style: the GCN weight is updated per snapshot via a matrix
    GRU, and each snapshot's projection uses its own evolved weight.

    Because the projection weight changes per snapshot, MatMul reuse via
    unique_gemm does not apply; only aggregation (IncreConv) is reused.
    """

    def __init__(self, in_feats, n_hidden, n_classes, classifier_hidden=307):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.W_init = Parameter(torch.Tensor(in_feats, n_hidden))
        init.xavier_uniform_(self.W_init)
        self.matgru = _MatGRUCell(in_feats, n_hidden)
        self.rgcn = IntraReuseConv(in_feats, n_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(n_hidden, classifier_hidden),
            nn.ReLU(),
            nn.Linear(classifier_hidden, n_classes),
        )

    def _evolve_and_project(self, aggregations):
        """Apply the per-snapshot evolved weight to each aggregation."""
        W = self.W_init
        gemm_out = None
        for agg in aggregations:
            W = self.matgru(W)
            gemm_out = torch.matmul(agg, W)
        return self.mlp(gemm_out)

    def forward_intra(self, g_list, changed_nodes, feat):
        aggregations = _aggregate_group_gcn(self.rgcn, g_list, feat)
        return self._evolve_and_project(aggregations)

    def forward_intra_3stream(self, g_list, changed_nodes, feat, rank):
        # The Conv stream handles IncreConv; the evolved-weight MatMul
        # cannot be fused into unique_gemm. For pipelining we reuse the GCN
        # pipeline helper to produce aggregations, then apply evolution
        # serially at the end.
        # To keep the pipeline meaningful we stream HtoD + Conv and collect
        # the per-snapshot aggregations directly.
        copy_stream = torch.cuda.Stream(priority=0)
        conv_stream = torch.cuda.Stream(priority=0)

        with torch.cuda.stream(copy_stream):
            g_dev = g_list[0].pin_memory_().to(rank, non_blocking=True)
            copy_done = [torch.cuda.Event()]
            copy_done[0].record(copy_stream)

        aggregations = []
        g_next_dev = None
        for i in range(len(g_list)):
            if i + 1 < len(g_list):
                with torch.cuda.stream(copy_stream):
                    g_next_dev = g_list[i + 1].pin_memory_().to(rank, non_blocking=True)
                    ev_cp = torch.cuda.Event()
                    ev_cp.record(copy_stream)
                    copy_done.append(ev_cp)

            conv_stream.wait_event(copy_done[i])
            with torch.cuda.stream(conv_stream):
                if i == 0:
                    aggregations.append(self.rgcn(g_dev, feat))
                else:
                    aggregations.append(self.rgcn.incre_aggregate(
                        feat, aggregations[-1], g_dev))

            if i + 1 < len(g_list):
                g_dev = g_next_dev

        torch.cuda.synchronize()
        return self._evolve_and_project(aggregations)

    def forward_inter_save(self, g_list, feat, changed_nodes, inter_reuse_idx):
        aggregations = _aggregate_group_gcn(self.rgcn, g_list, feat)
        cached = [aggregations[i] for i in inter_reuse_idx]
        return self._evolve_and_project(aggregations), cached

    def forward_inter_apply(self, g_list, feat, changed_nodes,
                            inter_reuse_idx, cached_aggregations):
        if not inter_reuse_idx:
            return self.forward_intra(g_list, changed_nodes, feat)
        aggregations = [None] * len(g_list)
        cached_iter = iter(cached_aggregations)
        for i in range(len(g_list)):
            if i in inter_reuse_idx:
                aggregations[i] = next(cached_iter)
                continue
            if i == 0:
                aggregations[i] = self.rgcn(g_list[i], feat)
            else:
                aggregations[i] = self.rgcn.incre_aggregate(
                    feat, aggregations[i - 1], g_list[i])
        return self._evolve_and_project(aggregations)


# =============================================================================
# GATLSTM: multi-head GAT + LSTM.
# =============================================================================

class GATLSTM(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.rgat = IntraReuseGAT(in_feats, hidden_size, num_heads)
        self._rnn = nn.LSTM(hidden_size * num_heads, hidden_size,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, out_feats)

    # --- temporal ---------------------------------------------------------

    def _temporal(self, per_snapshot_feats):
        flat = [h.flatten(1) for h in per_snapshot_feats]
        last, _ = _rnn_last_step(self._rnn, flat)
        return self.fc(last)

    # --- intra ------------------------------------------------------------

    def _aggregate_group_gat(self, g_list, feat):
        """Run the GAT stage with Type-I reuse on a snapshot group."""
        outs = []
        reuse_cache = None
        for i, g in enumerate(g_list):
            if i == 0:
                out, reuse_cache = self.rgat(g, feat)
            else:
                # The changed-node set is needed by increGAT_forward so that
                # only touched rows consume fresh attention-weighted messages
                # and unchanged rows are carried over from the cache.
                out = self.rgat.incre_forward(
                    g, reuse_cache, outs[-1], changed_nodes=None)
            outs.append(out)
        return outs, reuse_cache

    def forward_intra(self, g_list, changed_nodes, feat):
        outs, _ = self._aggregate_group_gat(g_list, feat)
        return self._temporal(outs)

    def forward_intra_3stream(self, g_list, changed_nodes, feat, rank):
        outs = triple_stream_pipeline_gat(
            self.rgat, g_list, changed_nodes, feat, rank)
        return self._temporal(outs)

    # --- inter ------------------------------------------------------------

    def forward_inter_save(self, g_list, feat, changed_nodes, inter_reuse_idx):
        outs, _ = self._aggregate_group_gat(g_list, feat)
        cached = [outs[i].clone().detach() for i in inter_reuse_idx]
        return self._temporal(outs), cached

    def forward_inter_apply(self, g_list, feat, changed_nodes,
                            inter_reuse_idx, cached_outs):
        if not inter_reuse_idx:
            return self.forward_intra(g_list, changed_nodes, feat)
        outs = []
        reuse_cache = None
        cached_iter = iter(cached_outs)
        for i, g in enumerate(g_list):
            if i in inter_reuse_idx:
                outs.append(next(cached_iter).clone())
                continue
            if i == 0 or reuse_cache is None:
                out, reuse_cache = self.rgat(g, feat)
            else:
                out = self.rgat.incre_forward(
                    g, reuse_cache, outs[-1], changed_nodes=None)
            outs.append(out)
        return self._temporal(outs)
