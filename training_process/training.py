"""
Distributed training driver for the reuse-aware dynamic GNN model.

Each training iteration consumes a snapshot-group assignment produced by the
upstream scheduler (ILP or greedy). The scheduler emits two kinds of slots:

  * Singleton slot: one snapshot group on this GPU in this iteration.
    Executed with ``forward_intra_3stream``, which combines Type-I
    (snapshot-level) reuse with the triple-stream pipeline (HtoD / Conv /
    MatMul).

  * Paired slot: two consecutive snapshot groups co-located on this GPU,
    with a known overlap set between them. Executed as a save/apply pair:
    the first group runs ``forward_inter_save`` and caches the shared
    aggregations; the second group runs ``forward_inter_apply`` and consumes
    them, avoiding redundant HtoD + GraphConv for the overlap.

Both code paths benefit from Type-I (intra) reuse; paired slots additionally
benefit from Type-II (inter) reuse.
"""

import argparse
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import WDGCN, TGCN, EvolveGCN, GATLSTM
from mydataloader import RealDataLoader
from tools import read_strategy_file


# Per-dataset input / output dimensions.
_DATASET_DIMS = {
    "arxiv":              (128, 40),
    "products":           (100, 47),
    "reddit":             (602, 41),
    "stackoverflow":      (128,  2),
    "stackoverflow-a2q":  (128,  2),
    "wiki":               (128,  2),
}

# Per-line number of the strategy file that stores each model's schedule.
_MODEL_STRATEGY_LINE = {"evolvegcn": 2, "wdgcn": 4, "tgcn": 6, "gatlstm": 8}


def _build_model(model_name, dataset, hidden_size, rank):
    in_feats, out_feats = _DATASET_DIMS[dataset]
    kwargs = dict(in_feats=in_feats, hidden_size=hidden_size, out_feats=out_feats)
    if model_name == "wdgcn":
        model = WDGCN(**kwargs)
    elif model_name == "tgcn":
        model = TGCN(**kwargs)
    elif model_name == "evolvegcn":
        # EvolveGCN uses slightly different hyperparameter names.
        model = EvolveGCN(in_feats=in_feats,
                          n_hidden=hidden_size,
                          n_classes=out_feats)
    elif model_name == "gatlstm":
        model = GATLSTM(**kwargs, num_heads=1)
    else:
        raise ValueError(f"unknown model: {model_name}")
    model = model.to(rank)
    return nn.parallel.DistributedDataParallel(model, device_ids=[rank])


def _run_singleton_slot(model, optimizer, g_list, changed_nodes, feat, label,
                        rank, use_3stream):
    """Singleton slot: Type-I intra-group reuse (optionally triple-streamed)."""
    g_list = [g.to(rank) for g in g_list]
    optimizer.zero_grad()
    with model.no_sync():
        if use_3stream:
            predictions = model.module.forward_intra_3stream(
                g_list, changed_nodes, feat, rank)
        else:
            predictions = model.module.forward_intra(
                g_list, changed_nodes, feat)
        loss = F.cross_entropy(predictions, label)
        loss.backward()
        optimizer.step()
    return loss.item()


def _move_paired_groups_to_gpu(g_lists, inter_indices, rank):
    """
    Move a paired slot's two snapshot lists to GPU, sharing graphs at the
    overlap positions so inter-group reuse can skip redundant HtoD transfers.

    Args:
        g_lists:       [g_list_a, g_list_b], two consecutive groups on CPU.
        inter_indices: (idx_a, idx_b) lists; positions in group a / b that
                       correspond to the same snapshots.
        rank:          target GPU id.

    Returns:
        [g_list_a_gpu, g_list_b_gpu] with shared graph objects at overlap.
    """
    idx_a, idx_b = inter_indices
    gpu_a, gpu_b = [], []

    if not idx_a:
        gpu_a = [g.to(rank) for g in g_lists[0]]
        gpu_b = [g.to(rank) for g in g_lists[1]]
        return [gpu_a, gpu_b]

    # Transfer group-a graphs and remember the ones that overlap with b.
    overlap = {}
    for j, g in enumerate(g_lists[0]):
        g_dev = g.to(rank)
        gpu_a.append(g_dev)
        if j in idx_a:
            overlap[j] = g_dev

    # Group-b reuses the overlapping GPU graphs; the rest are transferred.
    for j, g in enumerate(g_lists[1]):
        if j in idx_b:
            # Shared with group a at the matching position.
            match = idx_a[idx_b.index(j)]
            gpu_b.append(overlap[match])
        else:
            gpu_b.append(g.to(rank))
    return [gpu_a, gpu_b]


def _run_paired_slot(model, optimizer, g_lists, changed_nodes, inter_indices,
                     feat, label, rank):
    """Paired slot: Type-I + Type-II reuse via save/apply."""
    gpu_lists = _move_paired_groups_to_gpu(g_lists, inter_indices, rank)

    # First group: save the aggregations at the overlap positions.
    optimizer.zero_grad()
    with model.no_sync():
        predictions, saved = model.module.forward_inter_save(
            gpu_lists[0], feat, changed_nodes[0], inter_indices[0])
        loss_a = F.cross_entropy(predictions, label)
        loss_a.backward(retain_graph=True)
        optimizer.step()

        # Second group: consume the saved aggregations.
        optimizer.zero_grad()
        predictions = model.module.forward_inter_apply(
            gpu_lists[1], feat, changed_nodes[1], inter_indices[1], saved)
        loss_b = F.cross_entropy(predictions, label)
        loss_b.backward()
        optimizer.step()
    return loss_b.item()


def train(rank, dataset, model_name, world_size, time_window_size, num_epochs,
          strategy_path, hidden_size, use_3stream):
    print(f"[rank {rank}] training model={model_name} dataset={dataset}, "
          f"window={time_window_size}, 3stream={use_3stream}")
    dist.init_process_group("nccl", "tcp://127.0.0.1:12347",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    # Load schedule + snapshot groups.
    loader = RealDataLoader(rank, world_size, dataset, time_window_size)
    strategy = read_strategy_file(
        strategy_path, line_number=_MODEL_STRATEGY_LINE[model_name])
    groups, changed_node_list, inter_reuse_list, label, feat = \
        loader.load_graph_with_strategy(strategy)
    label = label.to(rank)
    feat = feat.to(rank)

    model = _build_model(model_name, dataset, hidden_size, rank)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    per_epoch_ms = []
    for epoch in range(num_epochs):
        t0 = time.time()
        for it, g_lists in enumerate(groups):
            if len(g_lists) == 1:
                loss = _run_singleton_slot(
                    model, optimizer,
                    g_lists[0], changed_node_list[it], feat, label, rank,
                    use_3stream=use_3stream)
                tag = "single"
            else:
                loss = _run_paired_slot(
                    model, optimizer,
                    g_lists, changed_node_list[it], inter_reuse_list[it],
                    feat, label, rank)
                tag = "paired"
            print(f"[rank {rank}] epoch={epoch} iter={it} ({tag}) "
                  f"loss={loss:.4f}")
            dist.barrier()

        scheduler.step()
        per_epoch_ms.append(round(1000 * (time.time() - t0), 2))

    if rank == 0:
        print("per-epoch (ms):", per_epoch_ms)

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="arxiv",
                        choices=list(_DATASET_DIMS))
    parser.add_argument("--model", type=str, default="wdgcn",
                        choices=list(_MODEL_STRATEGY_LINE))
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--time_window_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--strategy_path", type=str,
                        default="YOUR_STRATEGY_FILE",
                        help="Text file containing the scheduler's output")
    parser.add_argument("--no_3stream", action="store_true",
                        help="Disable the triple-stream pipeline (intra-only)")
    args = parser.parse_args()

    print(f"Launching {args.world_size} GPUs")
    mp.spawn(
        train,
        args=(args.dataset, args.model, args.world_size, args.time_window_size,
              args.num_epochs, args.strategy_path, args.hidden_size,
              not args.no_3stream),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
