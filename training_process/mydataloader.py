"""
Snapshot-group dataset loader for reuse-aware DGNN training.

Given a schedule produced by the upstream scheduler (ILP or greedy), each
slot specifies either one snapshot group (singleton) or two consecutive
snapshot groups co-located on the same GPU (paired). For each paired slot
the loader additionally computes the overlap indices between the two groups
(Type-II group-level reuse positions).

Dataset-specific pieces (base features, labels, on-disk graph layout) are
factored into ``_DATASET_CONFIG`` so the core logic is dataset-agnostic.
Graph and DiffMap directories default to ``YOUR_GRAPH_ROOT/<dataset>/...``
and can be overridden via the constructor or the ``DGNN_GRAPH_ROOT``
environment variable.
"""

import os

import dgl
import torch
from dgl.data import RedditDataset
from ogb.nodeproppred import NodePropPredDataset


# =============================================================================
# Dataset configuration.
# =============================================================================

def _ogb_features(name):
    def _load(root):
        dataset = NodePropPredDataset(name=name, root=root)
        graph, labels = dataset[0]
        feat = torch.tensor(graph['node_feat'])
        label = torch.from_numpy(labels).squeeze()
        return feat, label
    return _load


def _reddit_features(_root):
    g = RedditDataset(self_loop=True)[0]
    return g.ndata['feat'], g.ndata['label']


def _synthetic_features(num_nodes, feat_dim=128, num_classes=2):
    def _load(_root):
        feat = torch.randn(num_nodes, feat_dim)
        label = torch.randint(low=0, high=num_classes,
                              size=(num_nodes,), dtype=torch.int64)
        return feat, label
    return _load


_DATASET_CONFIG = {
    "arxiv":             {"load_fn": _ogb_features("ogbn-arxiv")},
    "products":          {"load_fn": _ogb_features("ogbn-products")},
    "reddit":            {"load_fn": _reddit_features},
    "stackoverflow":     {"load_fn": _synthetic_features(2_601_977)},
    "stackoverflow-a2q": {"load_fn": _synthetic_features(2_464_606)},
    "wiki":              {"load_fn": _synthetic_features(1_140_149)},
}


# =============================================================================
# Helpers.
# =============================================================================

def _changed_node_indices(graph):
    """Extract the changed-node index array from a DiffMap graph."""
    return torch.nonzero(graph.ndata['changed'],
                         as_tuple=False).squeeze().numpy()


def _load_group_snapshots(start_idx, time_window_size, graph_dir, diffmap_dir):
    """
    Load one snapshot group: the base snapshot plus ``time_window_size - 1``
    consecutive DiffMaps.

    Files on disk are named:
        <graph_dir>/Snapshot_XXX.dgl
        <diffmap_dir>/DiffMap_XXX.dgl
    """
    filepaths = [os.path.join(graph_dir, f"Snapshot_{start_idx:03d}.dgl")]
    filepaths += [os.path.join(diffmap_dir, f"DiffMap_{start_idx + j:03d}.dgl")
                  for j in range(1, time_window_size)]
    graphs = []
    for f in filepaths:
        g_list, _ = dgl.load_graphs(f)
        graphs.append(g_list[0])
    return graphs


def _compute_overlap(slot, time_window_size):
    """
    For a paired slot ``[s1, s2]`` (s1 < s2, each a snapshot-group start
    index), return the position indices inside each group of the snapshots
    shared between the two groups.

    For a singleton slot, return empty lists.
    """
    if len(slot) != 2:
        return []
    s1, s2 = slot
    overlap = set(range(s1, s1 + time_window_size)) & \
              set(range(s2, s2 + time_window_size))
    idx_a, idx_b = [], []
    for snap in sorted(overlap):
        idx_a.append(snap - s1)
        idx_b.append(snap - s2)
    return [idx_a, idx_b]


def _collect_changed_nodes(slot_groups):
    """
    For each slot (one or two groups), return the per-snapshot changed-node
    indices (extracted from each DiffMap's ``ndata['changed']`` mask). The
    base snapshot has no DiffMap, so position 0 is skipped.
    """
    if len(slot_groups) == 1:
        return [_changed_node_indices(g) for g in slot_groups[0][1:]]
    return [[_changed_node_indices(g) for g in slot_groups[0][1:]],
            [_changed_node_indices(g) for g in slot_groups[1][1:]]]


# =============================================================================
# Loader.
# =============================================================================

class RealDataLoader:
    """
    Loads per-slot snapshot groups (plus overlap metadata) required by
    reuse-aware training.

    Args:
        rank:             DDP rank; only the slots assigned to this rank are
                          loaded (round-robin over ``world_size``).
        world_size:       total number of processes.
        dataset:          one of ``_DATASET_CONFIG``.
        time_window_size: number of snapshots per group (base + DiffMaps).
        graph_root:       root directory containing per-dataset graph / diffmap
                          subtrees. Defaults to the ``DGNN_GRAPH_ROOT``
                          environment variable or ``YOUR_GRAPH_ROOT`` if unset.
        data_root:        root directory for base datasets (OGB cache).
    """

    def __init__(self, rank, world_size, dataset, time_window_size,
                 graph_root=None, data_root=None):
        if dataset not in _DATASET_CONFIG:
            raise ValueError(f"unknown dataset: {dataset}")
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.time_window_size = time_window_size
        self.graph_root = (graph_root
                           or os.environ.get('DGNN_GRAPH_ROOT')
                           or 'YOUR_GRAPH_ROOT')
        self.data_root = (data_root
                          or os.environ.get('DGNN_DATA_ROOT')
                          or 'YOUR_DATA_ROOT')

    def _graph_dir(self):
        return os.path.join(self.graph_root, self.dataset, 'graphs')

    def _diffmap_dir(self):
        return os.path.join(self.graph_root, self.dataset, 'diffmaps')

    def load_graph_with_strategy(self, strategy):
        """
        Execute the given schedule and materialize the per-slot snapshots.

        Args:
            strategy: list of slots; each slot is ``[a]`` (singleton) or
                      ``[a, b]`` (paired), where each integer is the starting
                      snapshot index of a group.

        Returns:
            groups:             list (per slot) of 1 or 2 snapshot groups on CPU.
            changed_node_list:  list (per slot) of per-snapshot changed-node
                                index arrays; for paired slots this is a pair
                                of lists.
            inter_reuse_list:   list (per slot) of ``[idx_a, idx_b]`` overlap
                                positions for Type-II reuse; empty for
                                singletons.
            label:              ground-truth node labels tensor.
            feat:               base node-feature tensor.
        """
        # Distribute the schedule across DDP ranks.
        slots = [slot for i, slot in enumerate(strategy)
                 if i % self.world_size == self.rank]
        print(f"[rank {self.rank}] slots: {slots}")

        feat, label = _DATASET_CONFIG[self.dataset]['load_fn'](self.data_root)
        graph_dir, diffmap_dir = self._graph_dir(), self._diffmap_dir()

        groups = []
        changed_node_list = []
        inter_reuse_list = []

        for slot in slots:
            slot_groups = [_load_group_snapshots(
                start, self.time_window_size, graph_dir, diffmap_dir)
                for start in slot]
            groups.append(slot_groups)
            changed_node_list.append(_collect_changed_nodes(slot_groups))
            inter_reuse_list.append(_compute_overlap(slot, self.time_window_size))

        print(f"[rank {self.rank}] loaded {len(groups)} slots")
        return groups, changed_node_list, inter_reuse_list, label, feat
