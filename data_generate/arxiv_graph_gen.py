import torch
import dgl
import random
from ogb.nodeproppred import NodePropPredDataset
import os

# Load the original graph
dataset = NodePropPredDataset(name='ogbn-arxiv', root='./data')
graph, labels = dataset[0]
g = dgl.graph((graph['edge_index'][0], graph['edge_index'][1]))
g = dgl.to_simple(g)
g.ndata['feat'] = torch.tensor(graph['node_feat'])
g.ndata['label'] = torch.from_numpy(labels).squeeze()

# Build edge list
edges = g.edges()
edge_list = list(zip(edges[0].tolist(), edges[1].tolist()))
total_edges = len(edge_list)

# Snapshot parameters
num_snapshots = 100
snapshots = []
prev_edges = set()
output_dir = "YOUR_ARXIV_OUTPUT_DIR"  # e.g. /path/to/graphs/arxiv/

for i in range(num_snapshots):
    # Edge ratio grows linearly from 50% to 90%
    ratio = 0.5 + 0.4 * (i / (num_snapshots - 1))
    curr_edge_count = int(total_edges * ratio)

    if i == 0:
        curr_edges = set(random.sample(edge_list, curr_edge_count))
    else:
        retain_count = int(curr_edge_count * 0.9)
        new_count = curr_edge_count - retain_count

        retained_edges = set(random.sample(prev_edges, min(retain_count, len(prev_edges))))
        remaining_pool = list(set(edge_list) - retained_edges)
        new_edges = set(random.sample(remaining_pool, new_count))

        curr_edges = retained_edges.union(new_edges)

    # Build DGLGraph snapshot
    src, dst = zip(*curr_edges)
    sg = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=g.num_nodes())
    # sg.ndata['feat'] = g.ndata['feat']
    # sg.ndata['label'] = g.ndata['label']
    snapshots.append(sg)
    prev_edges = curr_edges

    file_path = os.path.join(output_dir, f"Snapshot_{i:03d}.dgl")
    dgl.save_graphs(file_path, [sg])

    print(f"Snapshot {i+1:03d}: {sg.num_edges()} edges ({100 * ratio:.1f}% of original)")
