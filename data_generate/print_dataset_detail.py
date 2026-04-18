import os
import dgl
import torch

# Directory containing the snapshot .dgl files
graph_dir = "YOUR_GRAPHS_DIR"  # e.g. /path/to/graphs/wiki/graphs/

dgl_files = [
    os.path.join(graph_dir, f)
    for f in os.listdir(graph_dir)
    if f.endswith(".dgl")
]

total_edges = 0
total_nodes = 0
total_degree_sum = 0
graph_count = 0

for f in dgl_files:
    graphs, _ = dgl.load_graphs(f)
    g = graphs[0]  # each file contains exactly one snapshot
    num_edges = g.num_edges()
    num_nodes = g.num_nodes()

    degrees = g.in_degrees() + g.out_degrees()
    degree_sum = torch.sum(degrees).item()

    total_edges += num_edges
    total_nodes += num_nodes
    total_degree_sum += degree_sum
    graph_count += 1

if graph_count > 0:
    avg_edges = total_edges / graph_count
    avg_degree = total_degree_sum / total_nodes
    print(f"Average edges per graph: {avg_edges:.2f}")
    print(f"Average degree per node: {avg_degree:.2f}")
else:
    print("No .dgl files found.")
