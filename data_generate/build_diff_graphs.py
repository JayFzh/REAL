import torch
import dgl
import os
import numpy as np
import time

def load_snapshot_graphs(start_idx=0, end_idx=99, diff=False, graph_dir="YOUR_GRAPHS_DIR"):
    """
    Load snapshot graph files in index order from the given directory.

    Args:
        start_idx: first snapshot index to load (inclusive, default 0)
        end_idx:   last  snapshot index to load (inclusive, default 99)
        diff:      if True, load DiffMap_XXX.dgl files instead of Snapshot_XXX.dgl
        graph_dir: path to the directory containing the .dgl files

    Returns:
        list: graphs in index order
    """
    graphs = []

    for i in range(start_idx, end_idx + 1):
        filename = f"DiffMap_{i:03d}.dgl" if diff else f"Snapshot_{i:03d}.dgl"
        filepath = os.path.join(graph_dir, filename)

        if os.path.exists(filepath):
            print(f"Loading graph file: {filename}")
            graph_list, labels = dgl.load_graphs(filepath)

            if len(graph_list) > 0:
                graph = graph_list[0]
                graphs.append(graph)
                print(f"  Graph {i}: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
            else:
                print(f"  Warning: {filename} contains no graph data")
        else:
            print(f"  Error: file {filename} not found")

    print(f"\nLoaded {len(graphs)} graphs in total")
    return graphs


def load_all_arxiv_snapshots():
    """Load all Arxiv snapshots (Snapshot_000.dgl to Snapshot_099.dgl)."""
    return load_snapshot_graphs(0, 99, graph_dir="YOUR_ARXIV_GRAPHS_DIR")


def load_all_products_snapshots():
    """Load all Products snapshots (Snapshot_000.dgl to Snapshot_099.dgl)."""
    return load_snapshot_graphs(0, 99, graph_dir="YOUR_PRODUCTS_GRAPHS_DIR")


def load_all_reddit_snapshots():
    """Load all Reddit snapshots (Snapshot_000.dgl to Snapshot_099.dgl)."""
    return load_snapshot_graphs(0, 99, graph_dir="YOUR_REDDIT_GRAPHS_DIR")


def load_all_stackoverflow_snapshots():
    """Load all StackOverflow snapshots (Snapshot_000.dgl to Snapshot_099.dgl)."""
    return load_snapshot_graphs(0, 99, graph_dir="YOUR_STK_GRAPHS_DIR")


def load_all_snapshots(graph_dir="YOUR_GRAPHS_DIR"):
    """
    Load all snapshots (Snapshot_000.dgl to Snapshot_099.dgl) from graph_dir.

    Args:
        graph_dir: path to the directory containing the .dgl files

    Returns:
        list: all snapshot graphs in index order
    """
    return load_snapshot_graphs(0, 99, graph_dir=graph_dir)


def load_specified_snapshots(specified_indices, time_window_size, graph_dir="YOUR_GRAPHS_DIR"):
    """
    Load snapshot groups starting at the given indices.

    Each group consists of time_window_size consecutive snapshots beginning at
    the specified index.

    Args:
        specified_indices: list of starting snapshot indices
        time_window_size:  number of consecutive snapshots per group
        graph_dir:         path to the directory containing the .dgl files

    Returns:
        list[list]: each element is a list of DGL graphs for one group
    """
    groups = []

    for i in specified_indices:
        filenames = [f"Snapshot_{i:03d}.dgl"] + [f"Snapshot_{i+j:03d}.dgl" for j in range(1, time_window_size)]
        filepaths = [os.path.join(graph_dir, fn) for fn in filenames]
        group = []
        for f in filepaths:
            if os.path.exists(f):
                graph_list, labels = dgl.load_graphs(f)
                if len(graph_list) > 0:
                    group.append(graph_list[0])
                else:
                    print(f"  Warning: {f} contains no graph data")
            else:
                print(f"  Error: file {f} not found")
        groups.append(group)

    return groups


def load_specified_snapshots_diffmaps(specified_indices, time_window_size,
                                      graph_dir="YOUR_GRAPHS_DIR",
                                      diff_dir="YOUR_DIFFMAPS_DIR"):
    """
    Load snapshot groups together with their DiffMaps.

    For each starting index i, the group contains:
      - Snapshot_{i:03d}.dgl  (from graph_dir)
      - DiffMap_{i+1:03d}.dgl, DiffMap_{i+2:03d}.dgl, ... (from diff_dir)

    Args:
        specified_indices: list of starting snapshot indices
        time_window_size:  number of files per group (1 snapshot + (size-1) diffmaps)
        graph_dir:         path to the directory containing snapshot .dgl files
        diff_dir:          path to the directory containing diffmap .dgl files

    Returns:
        list[list]: each element is a list of DGL graphs for one group
    """
    groups = []

    for i in specified_indices:
        filenames = [f"Snapshot_{i:03d}.dgl"] + [f"DiffMap_{i+j:03d}.dgl" for j in range(1, time_window_size)]
        filepaths = [os.path.join(graph_dir, filenames[0])] + \
                    [os.path.join(diff_dir, fn) for fn in filenames[1:]]
        group = []
        for f in filepaths:
            if os.path.exists(f):
                graph_list, labels = dgl.load_graphs(f)
                if len(graph_list) > 0:
                    group.append(graph_list[0])
                else:
                    print(f"  Warning: {f} contains no graph data")
            else:
                print(f"  Error: file {f} not found")
        groups.append(group)

    return groups


def build_diff_map(G1, G2, directed=False):
    """
    Compute the differential map between two consecutive snapshot graphs.

    The DiffMap records added edges (weight=+1) and removed edges (weight=-1)
    relative to G1, and marks each node as changed or unchanged.
    It also stores the per-node in-degrees from G1 and G2 as
    ``prev_node_degree`` and ``curr_node_degree``, which are required by
    GraphConv's incremental renormalization (IncreConv).

    Args:
        G1:       source snapshot (DGL graph)
        G2:       target snapshot (DGL graph)
        directed: if False, normalize edge direction before comparison

    Returns:
        DGL graph with:
          edata['weight']           -- +1 for added edges, -1 for removed edges
          ndata['changed']          -- bool mask: True if node has any edge change
          ndata['unchanged']        -- bool mask: ~changed
          ndata['prev_node_degree'] -- float32, in-degree of each node in G1
          ndata['curr_node_degree'] -- float32, in-degree of each node in G2
    """
    edges1 = set(zip(G1.edges()[0].tolist(), G1.edges()[1].tolist()))
    edges2 = set(zip(G2.edges()[0].tolist(), G2.edges()[1].tolist()))

    if not directed:
        edges1 = set((min(u, v), max(u, v)) for (u, v) in edges1)
        edges2 = set((min(u, v), max(u, v)) for (u, v) in edges2)

    added_edges   = edges2 - edges1
    removed_edges = edges1 - edges2

    # Collect nodes touched by any edge change
    changed_nodes = set()
    for u, v in added_edges:
        changed_nodes.add(u); changed_nodes.add(v)
    for u, v in removed_edges:
        changed_nodes.add(u); changed_nodes.add(v)

    num_nodes = max(G1.num_nodes(), G2.num_nodes())
    all_nodes = set(range(num_nodes))
    unchanged_nodes = all_nodes - changed_nodes
    print(f"Changed nodes: {len(changed_nodes)}, Unchanged nodes: {len(unchanged_nodes)}")

    node_changed = torch.zeros(num_nodes, dtype=torch.bool)
    for node in changed_nodes:
        if node < num_nodes:
            node_changed[node] = True

    src, dst, weight = [], [], []
    for u, v in added_edges:
        src.append(u); dst.append(v); weight.append(1.0)
    for u, v in removed_edges:
        src.append(u); dst.append(v); weight.append(-1.0)

    if len(src) == 0:
        diff_map = dgl.graph(([], []), num_nodes=num_nodes)
        diff_map.edata['weight'] = torch.tensor([], dtype=torch.float32)
    else:
        diff_map = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=num_nodes)
        diff_map.edata['weight'] = torch.tensor(weight, dtype=torch.float32)

    diff_map.ndata['changed']   = node_changed
    diff_map.ndata['unchanged'] = ~node_changed

    # Per-node in-degree from G1 (prev) and G2 (curr), used by IncreConv renorm.
    # Nodes that only exist in the larger of the two graphs get degree 0.
    prev_deg = torch.zeros(num_nodes, dtype=torch.float32)
    curr_deg = torch.zeros(num_nodes, dtype=torch.float32)
    prev_deg[:G1.num_nodes()] = G1.in_degrees().float()
    curr_deg[:G2.num_nodes()] = G2.in_degrees().float()

    diff_map.ndata['prev_node_degree'] = prev_deg
    diff_map.ndata['curr_node_degree'] = curr_deg

    return diff_map


def build_and_save_all_diff_maps(input_dir="YOUR_GRAPHS_DIR", output_dir="YOUR_DIFFMAPS_DIR"):
    """
    Load all snapshots, compute pairwise DiffMaps between consecutive snapshots,
    and save each DiffMap to output_dir.

    DiffMap_{i+1:03d}.dgl represents the structural change from Snapshot_i to Snapshot_{i+1}.

    Args:
        input_dir:  directory containing snapshot .dgl files
        output_dir: directory where DiffMap .dgl files will be written
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading all snapshot graphs ...")
    graphs = load_all_snapshots(input_dir)

    if len(graphs) < 2:
        print("Error: at least 2 graphs are required to compute diff maps")
        return

    print(f"\nBuilding diff maps ...")

    for i in range(len(graphs) - 1):
        print(f"\nComputing diff map {i} -> {i+1}")

        diff_map = build_diff_map(graphs[i], graphs[i+1], directed=True)

        output_filename = f"DiffMap_{i+1:03d}.dgl"
        output_filepath = os.path.join(output_dir, output_filename)
        dgl.save_graphs(output_filepath, [diff_map])

        print(f"Saved: {output_filename}")
        print(f"  Edges: {diff_map.num_edges()}, Nodes: {diff_map.num_nodes()}")
        print(f"  Changed nodes:   {diff_map.ndata['changed'].sum().item()}")
        print(f"  Unchanged nodes: {diff_map.ndata['unchanged'].sum().item()}")
        print(f"  prev_degree mean: {diff_map.ndata['prev_node_degree'].mean():.2f}, "
              f"curr_degree mean: {diff_map.ndata['curr_node_degree'].mean():.2f}")

    print(f"\nAll diff maps saved to: {output_dir}")
    print(f"Total diff maps generated: {len(graphs) - 1}")


if __name__ == "__main__":
    input_dir  = "YOUR_GRAPHS_DIR"   # e.g. /path/to/graphs/wiki/graphs/
    output_dir = "YOUR_DIFFMAPS_DIR" # e.g. /path/to/graphs/wiki/diffmaps/
    build_and_save_all_diff_maps(input_dir, output_dir)
