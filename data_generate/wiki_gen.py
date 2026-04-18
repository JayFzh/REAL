import numpy as np
import dgl
import os

# Parameters
DATA_PATH = "YOUR_WIKI_DATA_PATH"    # e.g. /path/to/wiki-talk-temporal.txt
OUTPUT_PATH = "YOUR_WIKI_OUTPUT_PATH"  # e.g. /path/to/org_snapshots/wiki.dgl

# 1) Load raw data
print("Loading raw data ...")
data = np.loadtxt(DATA_PATH, dtype=np.int64)
src = data[:, 0]
dst = data[:, 1]
ts  = data[:, 2]  # timestamps are loaded but not used for snapshot splitting here

# 2) Remap sparse node IDs to contiguous indices
print("Remapping sparse node IDs ...")
unique_nodes = np.unique(np.concatenate([src, dst]))
num_nodes = len(unique_nodes)
print(f"Found {num_nodes} unique nodes.")

node2newid = { nid: idx for idx, nid in enumerate(unique_nodes) }

print("Applying ID remapping ...")
map_func = np.vectorize(lambda x: node2newid[x])
new_src = map_func(src)
new_dst = map_func(dst)

# 3) Build a single DGL graph containing all edges
print("Building a single DGL graph with all edges ...")
full_graph = dgl.graph((new_src, new_dst), num_nodes=num_nodes)
print(f"Full graph constructed: {full_graph.num_nodes()} nodes, {full_graph.num_edges()} edges")

# 4) Serialize the full graph
print(f"Saving the full graph to {OUTPUT_PATH} ...")
output_dir = os.path.dirname(OUTPUT_PATH)
os.makedirs(output_dir, exist_ok=True)
dgl.save_graphs(OUTPUT_PATH, [full_graph])

print("All done.")
