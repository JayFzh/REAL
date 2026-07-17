"""
Profiler for reuse-aware DGNN scheduling.

Produces the *reuse log* consumed by the ILP / greedy schedulers
(`ILP_Greedy_solver/*.read_from_file`). For a chosen set of snapshot groups it
emits:

  * ``time_costs[k]`` -- the standalone (unoverlapped) execution time of group
    ``k``: the sum of every snapshot's HtoD + Conv, plus the group-level MatMul,
    RNN, and backward time.

  * ``reuse[i][j]`` (i < j, upper-triangular) -- the time saved when groups
    ``i`` and ``j`` are co-located on the same GPU in the same iteration
    (Type-II / inter-group reuse). Two groups reuse the snapshots they overlap
    in; the later-starting group holds that overlap as its prefix and skips the
    corresponding **HtoD + first-layer Conv**, which is exactly this saving.

Timing methodology (per the paper's Profiler): each phase is measured with CUDA
events (``event.elapsed_time``); the group is profiled over ``repeats`` runs
after ``warmup`` warmup runs, and the **median** is used to reject
runtime-noise outliers. All times are in milliseconds.

Because ``time_costs`` and ``reuse`` are both expressed in the same unoverlapped
phase-time units, the scheduler's ``L_{i,j} = sum t_k - sum R`` load model is
internally consistent (the triple-stream overlap is a separate runtime
optimization and is intentionally not folded into these analytic costs).

Output (matches the solver's ``read_from_file`` format)::

    [t_0, t_1, ..., t_{n-1}]

    [[0, R_01, R_02, ...], [0, 0, R_12, ...], ...]

A companion ``<out>.groups.txt`` records the group start indices so the solver's
group indices (0..n-1) can be mapped back to snapshot start indices when
assembling the training strategy file.

Example::

    python profiler.py --dataset arxiv --model wdgcn --window 4 \
        --num-snapshots 100 --stride 2 --repeats 20 --warmup 5 \
        --graph-root /path/to/graphs --data-root /path/to/data \
        --out arxiv_wdgcn_reuse.txt
"""

import argparse
import os
import statistics

import torch
import torch.nn.functional as F

from model import (WDGCN, TGCN, EvolveGCN, GATLSTM,
                   _project_with_unique_gemm)
from mydataloader import (_DATASET_CONFIG, _load_group_snapshots,
                          _changed_node_indices)


# Per-dataset (in_feats, out_feats); mirrors training.py._DATASET_DIMS.
_DATASET_DIMS = {
    "arxiv":              (128, 40),
    "products":           (100, 47),
    "reddit":             (602, 41),
    "stackoverflow":      (128,  2),
    "stackoverflow-a2q":  (128,  2),
    "wiki":               (128,  2),
}


def _build_model(model_name, dataset, hidden_size, rank):
    in_feats, out_feats = _DATASET_DIMS[dataset]
    if model_name == "wdgcn":
        model = WDGCN(in_feats, hidden_size, out_feats)
    elif model_name == "tgcn":
        model = TGCN(in_feats, hidden_size, out_feats)
    elif model_name == "evolvegcn":
        model = EvolveGCN(in_feats=in_feats, n_hidden=hidden_size,
                          n_classes=out_feats)
    elif model_name == "gatlstm":
        model = GATLSTM(in_feats, hidden_size, out_feats, num_heads=1)
    else:
        raise ValueError(f"unknown model: {model_name}")
    return model.to(rank)


def _event_time(fn):
    """Run ``fn`` and return ``(result, elapsed_ms)`` measured with CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end)


def _profile_once(model, model_name, graphs, changed_nodes, feat, label, rank):
    """One phase-timed forward + backward over a snapshot group.

    Returns ``(htod_per_snapshot, conv_per_snapshot, tail_ms)`` where ``tail_ms``
    is the group-level MatMul + RNN + backward time.
    """
    w = len(graphs)
    htod = [0.0] * w
    conv = [0.0] * w
    model.zero_grad(set_to_none=True)

    if model_name == "gatlstm":
        outs = []
        cache = None
        for p, g in enumerate(graphs):
            g_dev, htod[p] = _event_time(lambda g=g: g.to(rank))
            if p == 0:
                (out, cache), conv[p] = _event_time(
                    lambda g_dev=g_dev: model.rgat(g_dev, feat))
            else:
                (out, cache), conv[p] = _event_time(
                    lambda g_dev=g_dev, prev=outs[-1], c=cache:
                    model.rgat.incre_forward(g_dev, c, prev, None))
            outs.append(out)

        def tail():
            pred = model._temporal(outs)
            loss = F.cross_entropy(pred, label)
            loss.backward()
            return loss
        _, tail_ms = _event_time(tail)
        return htod, conv, tail_ms

    # GCN-based (WDGCN / TGCN / EvolveGCN): decoupled aggregation + projection.
    aggs = []
    for p, g in enumerate(graphs):
        g_dev, htod[p] = _event_time(lambda g=g: g.to(rank))
        if p == 0:
            a, conv[p] = _event_time(
                lambda g_dev=g_dev: model.rgcn(g_dev, feat))
        else:
            a, conv[p] = _event_time(
                lambda g_dev=g_dev, prev=aggs[-1]:
                model.rgcn.incre_aggregate(feat, prev, g_dev))
        aggs.append(a)

    if model_name == "evolvegcn":
        def tail():
            pred = model._evolve_and_project(aggs)
            loss = F.cross_entropy(pred, label)
            loss.backward()
            return loss
    else:
        def tail():
            outs = _project_with_unique_gemm(model.rgcn, aggs, changed_nodes)
            pred = model._temporal(outs)
            loss = F.cross_entropy(pred, label)
            loss.backward()
            return loss
    _, tail_ms = _event_time(tail)
    return htod, conv, tail_ms


def profile_group(model, model_name, graphs, changed_nodes, feat, label,
                  rank, repeats, warmup):
    """Profile one group ``repeats`` times after ``warmup`` runs.

    Returns ``(t_k, htod_median[list], conv_median[list])`` using the per-phase
    median across the measured runs to reject outliers.
    """
    htod_runs, conv_runs, tail_runs = [], [], []
    for r in range(warmup + repeats):
        htod, conv, tail = _profile_once(
            model, model_name, graphs, changed_nodes, feat, label, rank)
        if r >= warmup:
            htod_runs.append(htod)
            conv_runs.append(conv)
            tail_runs.append(tail)

    w = len(graphs)
    htod_med = [statistics.median([run[p] for run in htod_runs])
                for p in range(w)]
    conv_med = [statistics.median([run[p] for run in conv_runs])
                for p in range(w)]
    tail_med = statistics.median(tail_runs)
    t_k = sum(htod_med) + sum(conv_med) + tail_med
    return t_k, htod_med, conv_med


def build_reuse_matrix(starts, window, htod_by_group, conv_by_group):
    """Estimate R[i][j] = HtoD + Conv of the snapshots groups i and j overlap in
    (the later-starting group holds the overlap as its prefix and skips it)."""
    n = len(starts)
    reuse = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = starts[i], starts[j]
            overlap = max(0, min(si + window, sj + window) - max(si, sj))
            if overlap == 0:
                continue
            # The later-starting group carries the overlap as positions 0..o-1.
            later = i if si > sj else j
            r = sum(htod_by_group[later][p] + conv_by_group[later][p]
                    for p in range(overlap))
            reuse[i][j] = round(r, 4)
    return reuse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(_DATASET_DIMS))
    ap.add_argument("--model", required=True,
                    choices=["wdgcn", "tgcn", "evolvegcn", "gatlstm"])
    ap.add_argument("--window", type=int, default=4,
                    help="snapshots per group (base + DiffMaps)")
    ap.add_argument("--starts", type=str, default=None,
                    help="comma-separated group start indices; if omitted, "
                         "sliding windows are generated from --num-snapshots/--stride")
    ap.add_argument("--num-snapshots", type=int, default=100)
    ap.add_argument("--stride", type=int, default=1,
                    help="stride between auto-generated sliding-window group starts "
                         "(stride < window yields overlapping groups with reuse)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--graph-root", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="reuse_log.txt")
    args = ap.parse_args()

    rank = args.device
    torch.cuda.set_device(rank)

    graph_root = (args.graph_root or os.environ.get("DGNN_GRAPH_ROOT")
                  or "YOUR_GRAPH_ROOT")
    data_root = (args.data_root or os.environ.get("DGNN_DATA_ROOT")
                 or "YOUR_DATA_ROOT")
    graph_dir = os.path.join(graph_root, args.dataset, "graphs")
    diffmap_dir = os.path.join(graph_root, args.dataset, "diffmaps")

    if args.starts:
        starts = [int(x) for x in args.starts.split(",")]
    else:
        starts = list(range(0, args.num_snapshots - args.window + 1, args.stride))
    if not starts:
        raise ValueError("no groups to profile (check --num-snapshots/--window/--stride)")

    feat, label = _DATASET_CONFIG[args.dataset]["load_fn"](data_root)
    feat = feat.float().to(rank)
    label = label.long().to(rank)
    model = _build_model(args.model, args.dataset, args.hidden, rank)

    n = len(starts)
    time_costs = [0.0] * n
    htod_by_group = [None] * n
    conv_by_group = [None] * n

    print(f"Profiling {n} groups (model={args.model}, dataset={args.dataset}, "
          f"window={args.window}, repeats={args.repeats}, warmup={args.warmup})")
    for gi, start in enumerate(starts):
        graphs = _load_group_snapshots(start, args.window, graph_dir, diffmap_dir)
        # Changed-node indices per DiffMap (positions 1..w-1), on the device so
        # they can index the GPU aggregations during projection.
        changed = [torch.as_tensor(_changed_node_indices(g),
                                   dtype=torch.long, device=rank)
                   for g in graphs[1:]]
        t_k, htod_med, conv_med = profile_group(
            model, args.model, graphs, changed, feat, label,
            rank, args.repeats, args.warmup)
        time_costs[gi] = round(t_k, 4)
        htod_by_group[gi] = htod_med
        conv_by_group[gi] = conv_med
        print(f"  group {gi:3d} (start={start:3d}): t_k={t_k:8.3f} ms  "
              f"(HtoD={sum(htod_med):.2f}, Conv={sum(conv_med):.2f})")

    reuse = build_reuse_matrix(starts, args.window, htod_by_group, conv_by_group)

    with open(args.out, "w") as f:
        f.write(repr(time_costs) + "\n\n" + repr(reuse) + "\n")
    with open(args.out + ".groups.txt", "w") as f:
        # group index -> snapshot start index, for translating the solver's
        # output back into a training strategy file.
        f.write(repr(starts) + "\n")

    print(f"\nWrote reuse log to {args.out} ({n} groups); "
          f"group->start map in {args.out}.groups.txt")


if __name__ == "__main__":
    main()
