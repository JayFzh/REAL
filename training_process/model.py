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
import dgl
from dgl.nn.pytorch import GraphConv, GATConv


# =============================================================================
# Deeper spatio-temporal encoders.
#
# A layer is one (GNN, RNN) block. Reuse (IncreConv, selective MatMul, the
# triple-stream pipeline, and Type-II group reuse) applies only at the first
# block's GNN, where the input node features are time-invariant within a
# snapshot group. Higher blocks take the previous block's per-snapshot RNN
# output as input, which varies across snapshots, so their GNN is recomputed in
# full on the actual snapshot graph -- reconstructed here from the base snapshot
# and the subsequent DiffMaps (added edges carry weight +1, removed edges -1).
# =============================================================================

def _reconstruct_full_graph_seq(g_list, device):
    """Rebuild the full snapshot graph at each position of a group.

    ``g_list`` is ``[base_snapshot, dmap_1, ..., dmap_{T-1}]``; each DiffMap
    carries ``edata['weight']`` of +1 for added edges and -1 for removed edges.
    Applying a DiffMap to the running edge set yields the next full snapshot.
    Returns ``[G_0, ..., G_{T-1}]`` on ``device``.
    """
    base = g_list[0]
    n = base.num_nodes()
    su, sv = base.edges()
    codes = su.to(device).long() * n + sv.to(device).long()
    graphs = [base.to(device)]
    for dm in g_list[1:]:
        du, dv = dm.edges()
        e = du.to(device).long() * n + dv.to(device).long()
        w = dm.edata['weight'].to(device)
        codes = torch.unique(torch.cat([codes, e[w > 0]]))
        removed = e[w < 0]
        if removed.numel():
            codes = codes[~torch.isin(codes, removed)]
        graphs.append(dgl.graph((codes // n, codes % n), num_nodes=n))
    return graphs


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
        # before each block's LSTM.
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

def _rnn_apply(rnn, per_snapshot_feats, batch_nodes=4096 * 256):
    """Stack features into [N, T, F], run the RNN in node-batches, and return
    the full output sequence [N, T, F]. Intermediate (GNN, RNN) blocks pass this
    sequence to the next block; the final block uses only its last timestep."""
    seq = torch.stack(per_snapshot_feats, dim=1)
    out_chunks = []
    for start in range(0, seq.size(0), batch_nodes):
        out, _ = rnn(seq[start:start + batch_nodes])
        out_chunks.append(out)
    return torch.cat(out_chunks, dim=0)


# =============================================================================
# Mixin that provides the four reuse entry points for GCN-based RNN models.
# =============================================================================

class _GCNReuseMixin:
    """
    Subclasses must provide:
        self.rgcn     : IntraReuseConv (the first block's GNN, reuse-aware)
        self.gnns     : nn.ModuleList of GraphConv (blocks 2..k, full recompute)
        self.rnns     : nn.ModuleList of nn.LSTM / nn.GRU (one per block)
        self._head(x) : takes [N, hidden] -> classifier logits
        self.num_layers
    """

    def _spatio_temporal(self, outs, g_list, device):
        """Stack of (GNN, RNN) blocks. ``outs`` is the first block's GNN output
        (from the reuse path); blocks 2..k recompute their GNN in full on the
        snapshots reconstructed from base+DiffMaps. Each block's RNN runs over
        the snapshot sequence -- intermediate blocks pass the full hidden-state
        sequence on to the next block, the last block feeds its final timestep
        to the classifier."""
        n = len(outs)
        fulls = None
        for b in range(self.num_layers):
            if b > 0:
                if fulls is None:
                    fulls = _reconstruct_full_graph_seq(g_list, device)
                conv = self.gnns[b - 1]
                outs = [conv(fulls[i], F.relu(outs[i])) for i in range(n)]
            seq = _rnn_apply(self.rnns[b], outs)
            if b < self.num_layers - 1:
                outs = [seq[:, t, :] for t in range(n)]
            else:
                return self._head(seq[:, -1, :])

    def forward_intra(self, g_list, changed_nodes, feat):
        aggregations = _aggregate_group_gcn(self.rgcn, g_list, feat)
        outs = _project_with_unique_gemm(self.rgcn, aggregations, changed_nodes)
        return self._spatio_temporal(outs, g_list, feat.device)

    def forward_intra_3stream(self, g_list, changed_nodes, feat, rank):
        outs, _ = _pipeline_gcn(self.rgcn, g_list, changed_nodes, feat, rank)
        return self._spatio_temporal(outs, g_list, rank)

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
        dev = rank if rank is not None else feat.device
        return self._spatio_temporal(outs, g_list, dev), saved

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
            outs = _project_with_unique_gemm(self.rgcn, aggregations, changed_nodes)
        dev = rank if rank is not None else feat.device
        return self._spatio_temporal(outs, g_list, dev)


# =============================================================================
# WDGCN: GCN + LSTM.
# =============================================================================

class WDGCN(nn.Module, _GCNReuseMixin):
    def __init__(self, in_feats, hidden_size, out_feats, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.rgcn = IntraReuseConv(in_feats, hidden_size)
        self.gnns = nn.ModuleList([
            GraphConv(hidden_size, hidden_size, norm='both',
                      activation=None, allow_zero_in_degree=True)
            for _ in range(num_layers - 1)])
        self.rnns = nn.ModuleList([
            nn.LSTM(hidden_size, hidden_size, batch_first=True)
            for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_size, out_feats)

    def _head(self, x):
        return self.fc(x)


# =============================================================================
# TGCN: GCN + GRU.
# =============================================================================

class TGCN(nn.Module, _GCNReuseMixin):
    def __init__(self, in_feats, hidden_size, out_feats, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.rgcn = IntraReuseConv(in_feats, hidden_size)
        self.gnns = nn.ModuleList([
            GraphConv(hidden_size, hidden_size, norm='both',
                      activation=None, allow_zero_in_degree=True)
            for _ in range(num_layers - 1)])
        self.rnns = nn.ModuleList([
            nn.GRU(hidden_size, hidden_size, batch_first=True)
            for _ in range(num_layers)])
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

    def __init__(self, in_feats, n_hidden, n_classes, classifier_hidden=307,
                 num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.rgcn = IntraReuseConv(in_feats, n_hidden)
        # Per-layer evolving weights (matrix GRU): layer 1 maps in_feats ->
        # n_hidden, higher layers n_hidden -> n_hidden.
        dims = [in_feats] + [n_hidden] * (num_layers - 1)
        self.W_inits = nn.ParameterList(
            [Parameter(torch.Tensor(d, n_hidden)) for d in dims])
        for w in self.W_inits:
            init.xavier_uniform_(w)
        self.matgrus = nn.ModuleList([_MatGRUCell(d, n_hidden) for d in dims])
        # Higher layers aggregate in full on the reconstructed snapshots; the
        # evolved weight (above) applies the projection, so these are weightless.
        self.gnns = nn.ModuleList([
            GraphConv(n_hidden, n_hidden, norm='both', weight=False, bias=False,
                      activation=None, allow_zero_in_degree=True)
            for _ in range(num_layers - 1)])
        self.mlp = nn.Sequential(
            nn.Linear(n_hidden, classifier_hidden),
            nn.ReLU(),
            nn.Linear(classifier_hidden, n_classes),
        )

    def _evolve_layer(self, aggregations, W_init, matgru):
        """One evolving-weight GCN layer: project each snapshot's aggregation
        with a weight the matrix GRU evolves across the snapshot sequence."""
        W = W_init
        outs = []
        for agg in aggregations:
            W = matgru(W)
            outs.append(torch.matmul(F.relu(agg), W))
        return outs

    def _stack(self, aggregations, g_list, device):
        """Stack of evolving-GCN layers. ``aggregations`` is the first layer's
        aggregation (from the reuse path); higher layers recompute aggregation
        in full on the reconstructed snapshots. Classify from the last snapshot,
        whose evolved weight has absorbed the full history."""
        h = self._evolve_layer(aggregations, self.W_inits[0], self.matgrus[0])
        fulls = None
        for b in range(1, self.num_layers):
            if fulls is None:
                fulls = _reconstruct_full_graph_seq(g_list, device)
            aggs = [self.gnns[b - 1](fulls[i], F.relu(h[i]))
                    for i in range(len(h))]
            h = self._evolve_layer(aggs, self.W_inits[b], self.matgrus[b])
        return self.mlp(h[-1])

    def forward_intra(self, g_list, changed_nodes, feat):
        aggregations = _aggregate_group_gcn(self.rgcn, g_list, feat)
        return self._stack(aggregations, g_list, feat.device)

    def forward_intra_3stream(self, g_list, changed_nodes, feat, rank):
        aggregations, _ = _pipeline_evolve_aggregate(
            self.rgcn, g_list, feat, rank)
        return self._stack(aggregations, g_list, rank)

    def forward_inter_save(self, g_list, feat, changed_nodes, inter_reuse_idx,
                           rank=None, use_3stream=False):
        if use_3stream:
            aggregations, saved = _pipeline_evolve_aggregate(
                self.rgcn, g_list, feat, rank, save_idx=inter_reuse_idx)
        else:
            aggregations = _aggregate_group_gcn(self.rgcn, g_list, feat)
            saved = [aggregations[i] for i in sorted(inter_reuse_idx)]
        dev = rank if rank is not None else feat.device
        return self._stack(aggregations, g_list, dev), saved

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
        dev = rank if rank is not None else feat.device
        return self._stack(aggregations, g_list, dev)


# =============================================================================
# GATLSTM: multi-head GAT + LSTM.
# =============================================================================

class GATLSTM(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, num_heads=4,
                 num_layers=1):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.rgat = IntraReuseGAT(in_feats, hidden_size, num_heads)
        # Higher-block GNNs: full GAT on the reconstructed snapshots, taking the
        # previous block's LSTM output and producing multi-head features.
        self.gnns = nn.ModuleList([
            GATConv(hidden_size, hidden_size, num_heads=num_heads,
                    activation=None, bias=False, residual=False,
                    allow_zero_in_degree=True)
            for _ in range(num_layers - 1)])
        self.rnns = nn.ModuleList([
            nn.LSTM(hidden_size * num_heads, hidden_size, batch_first=True)
            for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_size, out_feats)

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

    def _spatio_temporal(self, outs, g_list, device):
        """Stack of (GAT, LSTM) blocks. ``outs`` is the first block's GAT output
        (from the reuse path); higher blocks recompute the GAT in full on the
        reconstructed snapshots, taking the previous block's LSTM output. Each
        block's LSTM runs over the snapshot sequence; the last block feeds its
        final timestep to the classifier."""
        n = len(outs)
        fulls = None
        for b in range(self.num_layers):
            if b > 0:
                if fulls is None:
                    fulls = _reconstruct_full_graph_seq(g_list, device)
                gat = self.gnns[b - 1]
                outs = [gat(fulls[i], F.relu(outs[i])) for i in range(n)]
            flat = [F.relu(h).flatten(1) for h in outs]
            seq = _rnn_apply(self.rnns[b], flat)
            if b < self.num_layers - 1:
                outs = [seq[:, t, :] for t in range(n)]
            else:
                return self.fc(seq[:, -1, :])

    def forward_intra(self, g_list, changed_nodes, feat):
        outs, _ = self._aggregate_group_gat(g_list, feat)
        return self._spatio_temporal(outs, g_list, feat.device)

    def forward_intra_3stream(self, g_list, changed_nodes, feat, rank):
        outs, _ = _pipeline_gat(self.rgat, g_list, changed_nodes, feat, rank)
        return self._spatio_temporal(outs, g_list, rank)

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
        dev = rank if rank is not None else feat.device
        return self._spatio_temporal(outs, g_list, dev), saved

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
        else:
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
        dev = rank if rank is not None else feat.device
        return self._spatio_temporal(outs, g_list, dev)
