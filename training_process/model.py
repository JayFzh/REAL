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
        # activation=None: the reuse path caches/increments the PRE-activation
        # symmetric-norm aggregation (so the additive/rescale IncreConv identity
        # holds); ReLU is applied after the transformation (see
        # _project_with_unique_gemm / the pipeline), identically for every
        # snapshot in the group.
        self.gcn = GraphConv(
            in_feats, hidden_size,
            activation=None,
            norm='both', bias=False, weight=False,
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
        # activation=None: cache/increment the PRE-activation aggregation so the
        # softmax-denominator rescale identity holds; ReLU is applied uniformly
        # in GATLSTM._temporal before the RNN.
        self.gat = GATConv(
            in_feats, hidden_size,
            num_heads=num_heads,
            activation=None,
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
    per-snapshot outputs by cloning and scattering. ReLU is applied here (on
    the aggregation) so every snapshot receives the same nonlinearity.
    """
    shape_list = [aggregations[0].shape[0]]
    unique_feat = F.relu(aggregations[0])
    for i in range(1, len(aggregations)):
        changed = changed_nodes[i - 1]
        shape_list.append(len(changed))
        unique_feat = torch.cat((unique_feat, F.relu(aggregations[i][changed])), dim=0)
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
#
# A single unified pipeline per model family handles Type-I (intra) reuse, the
# HtoD/Conv/MatMul overlap, and -- when a `cached` prefix of aggregations from
# the previous group is supplied -- Type-II (inter) reuse, so all three compose
# in one code path.
#
# Graphs must arrive ON CPU (pinned): the copy stream performs the actual HtoD,
# overlapping it with Conv/MatMul. CUDA streams are created once and reused
# across iterations via _get_streams (avoids per-call stream construction).
# =============================================================================

_STREAM_POOL = {}


def _get_streams(device):
    """Return persistent (copy, conv, gemm) CUDA streams for ``device``.

    Streams are created once per device and reused across every pipeline call,
    rather than reconstructed each iteration.
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    key = (dev.type, dev.index)
    pool = _STREAM_POOL.get(key)
    if pool is None:
        pool = (
            torch.cuda.Stream(device=dev),            # copy  (HtoD)
            torch.cuda.Stream(device=dev),            # conv  (GraphConv / IncreConv)
            torch.cuda.Stream(device=dev, priority=-1),  # gemm (dense MatMul)
        )
        _STREAM_POOL[key] = pool
    return pool


def _to_device_async(g, rank, copy_stream):
    """Pinned, non-blocking HtoD of a CPU graph on the copy stream; returns the
    device graph and an event recorded after the transfer. Graphs are loaded
    once and reused across epochs, so pinning is guarded to a no-op if already
    pinned."""
    with torch.cuda.stream(copy_stream):
        if not g.is_pinned():
            g.pin_memory_()
        g_dev = g.to(rank, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(copy_stream)
    return g_dev, ev


def _pipeline_gcn(rgcn, g_list, changed_nodes, feat, rank,
                  cached=None, save_idx=None):
    """
    Triple-stream GCN pipeline with Type-I (intra) reuse and, when ``cached`` is
    given, Type-II (inter) reuse.

    Args:
        g_list:   snapshot group ON CPU (base snapshot + DiffMaps).
        cached:   ``{position: pre-activation aggregation}`` for overlap
                  snapshots reused from the previous group; their HtoD + Conv
                  are skipped (Type-II).
        save_idx: positions whose pre-activation aggregation to return for the
                  next group.

    Dependency per snapshot i: HtoD(i) -> Conv(i) -> MatMul(i); ReLU+MatMul run
    on the gemm stream so Conv(i+1) overlaps MatMul(i) and HtoD(i+2).

    Returns ``(outs, saved)``.
    """
    cached = cached or {}
    save_set = set(save_idx or [])
    copy_stream, conv_stream, gemm_stream = _get_streams(rank)
    n = len(g_list)

    conv_out_buf = [None] * n     # pre-activation aggregations (chaining + save)
    gemm_out_buf = [None] * n
    copy_done = {}
    conv_done = {}
    dev = {}

    def prefetch(i):
        if i in cached or i in dev:
            return
        dev[i], copy_done[i] = _to_device_async(g_list[i], rank, copy_stream)

    prefetch(0)
    for i in range(n):
        if i + 1 < n:
            prefetch(i + 1)

        # ---- Conv (or reuse the cached aggregation) ----
        if i in cached:
            conv_out = cached[i]
        else:
            conv_stream.wait_event(copy_done[i])
            with torch.cuda.stream(conv_stream):
                if i == 0:
                    conv_out = rgcn(dev[i], feat)
                else:
                    conv_out = rgcn.incre_aggregate(feat, conv_out_buf[i - 1], dev[i])
                ev = torch.cuda.Event()
                ev.record(conv_stream)
                conv_done[i] = ev
        conv_out_buf[i] = conv_out

        # ---- ReLU + MatMul on the gemm stream (gated on Conv(i)) ----
        if i in conv_done:
            gemm_stream.wait_event(conv_done[i])
        with torch.cuda.stream(gemm_stream):
            gin = F.relu(conv_out) if i == 0 else F.relu(conv_out[changed_nodes[i - 1]])
            gemm_out_buf[i] = rgcn.unique_gemm(gin)

    # The gemm stream is the tail of the HtoD->Conv->MatMul chain; syncing it
    # (rather than the whole device) makes every produced tensor safe to consume
    # on the default stream below and by the next group.
    gemm_stream.synchronize()

    outs = [gemm_out_buf[0]]
    for i in range(1, n):
        out = outs[-1].clone()
        out[changed_nodes[i - 1]] = gemm_out_buf[i]
        outs.append(out)

    saved = [conv_out_buf[i] for i in sorted(save_set)] if save_set else []
    return outs, saved


def _pipeline_gat(rgat, g_list, changed_nodes, feat, rank,
                  cached=None, save_idx=None):
    """
    Two-stream GAT pipeline (HtoD overlaps Conv; MatMul is cached inside the
    first full GAT call) with Type-I and optional Type-II reuse.

    ``cached`` maps ``{position: (out, reuse_cache)}`` for overlap snapshots:
    both the aggregation output and the attention intermediates (el, er,
    feat_src, node_denom) are reused, so the first non-overlap snapshot can be
    computed *incrementally* (increGAT_forward) rather than via a full GAT
    forward on a DiffMap. ``save_idx`` returns ``(out, reuse_cache)`` per
    position.
    """
    cached = cached or {}
    save_set = set(save_idx or [])
    copy_stream, conv_stream, _ = _get_streams(rank)
    n = len(g_list)

    outs = [None] * n
    caches = [None] * n
    copy_done = {}
    dev = {}
    reuse_cache = None

    def prefetch(i):
        if i in cached or i in dev:
            return
        dev[i], copy_done[i] = _to_device_async(g_list[i], rank, copy_stream)

    prefetch(0)
    for i in range(n):
        if i + 1 < n:
            prefetch(i + 1)

        if i in cached:
            out, reuse_cache = cached[i]
        else:
            conv_stream.wait_event(copy_done[i])
            with torch.cuda.stream(conv_stream):
                if reuse_cache is None:
                    # First materialized snapshot: full GAT forward on the full
                    # base graph (also produces the reusable attention cache).
                    out, reuse_cache = rgat(dev[i], feat)
                else:
                    out, reuse_cache = rgat.incre_forward(
                        dev[i], reuse_cache, outs[i - 1], changed_nodes=None)
        outs[i] = out
        caches[i] = reuse_cache

    conv_stream.synchronize()
    saved = [(outs[i], caches[i]) for i in sorted(save_set)] if save_set else []
    return outs, saved


def _pipeline_evolve_aggregate(rgcn, g_list, feat, rank,
                               cached=None, save_idx=None):
    """
    HtoD/Conv pipeline that produces the per-snapshot pre-activation
    aggregations for EvolveGCN (whose per-snapshot evolved-weight MatMul cannot
    be fused into ``unique_gemm``). Supports Type-II ``cached`` prefix and
    ``save_idx`` like the other pipelines.
    """
    cached = cached or {}
    save_set = set(save_idx or [])
    copy_stream, conv_stream, _ = _get_streams(rank)
    n = len(g_list)

    aggregations = [None] * n
    copy_done = {}
    dev = {}

    def prefetch(i):
        if i in cached or i in dev:
            return
        dev[i], copy_done[i] = _to_device_async(g_list[i], rank, copy_stream)

    prefetch(0)
    for i in range(n):
        if i + 1 < n:
            prefetch(i + 1)
        if i in cached:
            aggregations[i] = cached[i]
        else:
            conv_stream.wait_event(copy_done[i])
            with torch.cuda.stream(conv_stream):
                if i == 0:
                    aggregations[i] = rgcn(dev[i], feat)
                else:
                    aggregations[i] = rgcn.incre_aggregate(
                        feat, aggregations[i - 1], dev[i])

    conv_stream.synchronize()
    saved = [aggregations[i] for i in sorted(save_set)] if save_set else []
    return aggregations, saved


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
        outs, _ = _pipeline_gcn(self.rgcn, g_list, changed_nodes, feat, rank)
        return self._temporal(outs)

    def forward_inter_save(self, g_list, feat, changed_nodes, inter_reuse_idx,
                           rank=None, use_3stream=False):
        """First group of a paired slot: compute it (Type-I) and return the
        overlap aggregations for the second group (Type-II). With use_3stream
        the triple-stream pipeline is used, so Type-I + Type-II + 3-stream
        compose in one path."""
        if use_3stream:
            outs, saved = _pipeline_gcn(self.rgcn, g_list, changed_nodes, feat,
                                        rank, save_idx=inter_reuse_idx)
        else:
            aggregations = _aggregate_group_gcn(self.rgcn, g_list, feat)
            saved = [aggregations[i] for i in sorted(inter_reuse_idx)]
            outs = _project_with_unique_gemm(self.rgcn, aggregations, changed_nodes)
        return self._temporal(outs), saved

    def forward_inter_apply(self, g_list, feat, changed_nodes, inter_reuse_idx,
                            cached_aggregations, rank=None, use_3stream=False):
        """Second group: reuse the first group's overlap aggregations (Type-II)
        for the shared prefix; compute the rest incrementally."""
        if not inter_reuse_idx:
            if use_3stream:
                return self.forward_intra_3stream(g_list, changed_nodes, feat, rank)
            return self.forward_intra(g_list, changed_nodes, feat)
        cached = dict(zip(sorted(inter_reuse_idx), cached_aggregations))
        if use_3stream:
            outs, _ = _pipeline_gcn(self.rgcn, g_list, changed_nodes, feat, rank,
                                    cached=cached)
            return self._temporal(outs)
        aggregations = [None] * len(g_list)
        for i in range(len(g_list)):
            if i in cached:
                aggregations[i] = cached[i]
            elif i == 0:
                aggregations[i] = self.rgcn(g_list[i], feat)
            else:
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
            gemm_out = torch.matmul(F.relu(agg), W)
        return self.mlp(gemm_out)

    def forward_intra(self, g_list, changed_nodes, feat):
        aggregations = _aggregate_group_gcn(self.rgcn, g_list, feat)
        return self._evolve_and_project(aggregations)

    def forward_intra_3stream(self, g_list, changed_nodes, feat, rank):
        aggregations, _ = _pipeline_evolve_aggregate(
            self.rgcn, g_list, feat, rank)
        return self._evolve_and_project(aggregations)

    def forward_inter_save(self, g_list, feat, changed_nodes, inter_reuse_idx,
                           rank=None, use_3stream=False):
        if use_3stream:
            aggregations, saved = _pipeline_evolve_aggregate(
                self.rgcn, g_list, feat, rank, save_idx=inter_reuse_idx)
        else:
            aggregations = _aggregate_group_gcn(self.rgcn, g_list, feat)
            saved = [aggregations[i] for i in sorted(inter_reuse_idx)]
        return self._evolve_and_project(aggregations), saved

    def forward_inter_apply(self, g_list, feat, changed_nodes,
                            inter_reuse_idx, cached_aggregations,
                            rank=None, use_3stream=False):
        if not inter_reuse_idx:
            if use_3stream:
                return self.forward_intra_3stream(g_list, changed_nodes, feat, rank)
            return self.forward_intra(g_list, changed_nodes, feat)
        cached = dict(zip(sorted(inter_reuse_idx), cached_aggregations))
        if use_3stream:
            aggregations, _ = _pipeline_evolve_aggregate(
                self.rgcn, g_list, feat, rank, cached=cached)
        else:
            aggregations = [None] * len(g_list)
            for i in range(len(g_list)):
                if i in cached:
                    aggregations[i] = cached[i]
                elif i == 0:
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
        flat = [F.relu(h).flatten(1) for h in per_snapshot_feats]
        last, _ = _rnn_last_step(self._rnn, flat)
        return self.fc(last)

    # --- intra ------------------------------------------------------------

    def _aggregate_group_gat(self, g_list, feat):
        """Run the GAT stage with Type-I reuse; returns per-snapshot outputs and
        per-snapshot reuse caches (attention intermediates el/er/feat_src/denom)."""
        outs = []
        caches = []
        reuse_cache = None
        for i, g in enumerate(g_list):
            if i == 0:
                out, reuse_cache = self.rgat(g, feat)
            else:
                out, reuse_cache = self.rgat.incre_forward(
                    g, reuse_cache, outs[-1], changed_nodes=None)
            outs.append(out)
            caches.append(reuse_cache)
        return outs, caches

    def forward_intra(self, g_list, changed_nodes, feat):
        outs, _ = self._aggregate_group_gat(g_list, feat)
        return self._temporal(outs)

    def forward_intra_3stream(self, g_list, changed_nodes, feat, rank):
        outs, _ = _pipeline_gat(self.rgat, g_list, changed_nodes, feat, rank)
        return self._temporal(outs)

    # --- inter ------------------------------------------------------------

    def forward_inter_save(self, g_list, feat, changed_nodes, inter_reuse_idx,
                           rank=None, use_3stream=False):
        if use_3stream:
            outs, saved = _pipeline_gat(self.rgat, g_list, changed_nodes, feat,
                                        rank, save_idx=inter_reuse_idx)
        else:
            outs, caches = self._aggregate_group_gat(g_list, feat)
            # Cache both the output and the attention intermediates so the next
            # group can continue INCREMENTALLY from the overlap prefix.
            saved = [(outs[i], caches[i]) for i in sorted(inter_reuse_idx)]
        return self._temporal(outs), saved

    def forward_inter_apply(self, g_list, feat, changed_nodes, inter_reuse_idx,
                            cached, rank=None, use_3stream=False):
        if not inter_reuse_idx:
            if use_3stream:
                return self.forward_intra_3stream(g_list, changed_nodes, feat, rank)
            return self.forward_intra(g_list, changed_nodes, feat)
        cached_map = dict(zip(sorted(inter_reuse_idx), cached))
        if use_3stream:
            outs, _ = _pipeline_gat(self.rgat, g_list, changed_nodes, feat, rank,
                                    cached=cached_map)
            return self._temporal(outs)
        # Sequential (no 3-stream): reuse (out, reuse_cache) over the overlap
        # prefix, then continue incrementally over the suffix. This uses the
        # cached attention intermediates so the first suffix snapshot is an
        # incremental update rather than a (wrong) full forward on a DiffMap.
        outs = [None] * len(g_list)
        reuse_cache = None
        for i in range(len(g_list)):
            if i in cached_map:
                out, reuse_cache = cached_map[i]
            elif reuse_cache is None:
                out, reuse_cache = self.rgat(g_list[i], feat)
            else:
                out, reuse_cache = self.rgat.incre_forward(
                    g_list[i], reuse_cache, outs[i - 1], changed_nodes=None)
            outs[i] = out
        return self._temporal(outs)
