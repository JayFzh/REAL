import dgl
from dgl.data import RedditDataset
import torch

# Download and load the Reddit dataset
dataset = RedditDataset(self_loop=True)
g = dataset[0]

# Print graph info
print(g)
print("Node feature shape:", g.ndata['feat'].shape)
print("Label shape:", g.ndata['label'].shape)
