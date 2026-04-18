import dgl
from dgl.data import RedditDataset
import torch
import os

# Download and load the Reddit dataset
dataset = RedditDataset(self_loop=True)
g = dataset[0]

# Print graph info
print(g)
print("Node feature shape:", g.ndata['feat'].shape)
print("Label shape:", g.ndata['label'].shape)

# save_path = './data/reddit.dgl'
# os.makedirs(os.path.dirname(save_path), exist_ok=True)
# dgl.save_graphs(save_path, [g])
# print(f"Graph saved to {save_path}")
