import torch
import dgl
import random
import os
import numpy as np
from ogb.nodeproppred import NodePropPredDataset
import time

# Load the original graph
dataset = NodePropPredDataset(name='ogbn-products', root='./data')
graph, labels = dataset[0]
g = dgl.graph((graph['edge_index'][0], graph['edge_index'][1]))
g = dgl.to_simple(g)
g.ndata['feat'] = torch.tensor(graph['node_feat'])
g.ndata['label'] = torch.from_numpy(labels).squeeze()

src, dst = g.edges()
edge_array = torch.stack([src, dst], dim=1).cpu().numpy()
total_edges = len(edge_array)
print(f"Total edges in original graph: {total_edges}")

# Snapshot parameters
num_snapshots = 100
snapshots = []
prev_edge_idx = np.array([], dtype=np.int64)
output_dir = "YOUR_PRODUCTS_OUTPUT_DIR"  # e.g. /path/to/graphs/products/
os.makedirs(output_dir, exist_ok=True)

for i in range(num_snapshots):
    start = time.time()
    # Edge ratio grows linearly from 10% to 20%
    ratio = 0.1 + 0.1 * (i / (num_snapshots - 1))
    curr_edge_count = int(total_edges * ratio)

    if i == 0:
        print(f"Initial edge count: {curr_edge_count}")
        curr_edge_idx = np.random.choice(total_edges, size=curr_edge_count, replace=False)
    else:
        retain_count = int(curr_edge_count * 0.9)
        new_count = curr_edge_count - retain_count

        retain_count = min(retain_count, len(prev_edge_idx))
        retained = np.random.choice(prev_edge_idx, size=retain_count, replace=False)

        remaining_pool = np.setdiff1d(np.arange(total_edges), retained, assume_unique=True)
        new_edges = np.random.choice(remaining_pool, size=new_count, replace=False)

        curr_edge_idx = np.concatenate([retained, new_edges])

    # Index into edge_array
    curr_edges = edge_array[curr_edge_idx]
    sg = dgl.graph((torch.from_numpy(curr_edges[:,0]), torch.from_numpy(curr_edges[:,1])),
                   num_nodes=g.num_nodes())

    snapshots.append(sg)
    prev_edge_idx = curr_edge_idx

    file_path = os.path.join(output_dir, f"Snapshot_{i:03d}.dgl")
    dgl.save_graphs(file_path, [sg])

    print(f"Snapshot {i+1:03d}: {sg.num_nodes()} nodes {sg.num_edges()} edges ({100 * ratio:.1f}% of original)")
    print(f"Time taken for snapshot {i+1:03d}: {time.time() - start:.2f} seconds")
