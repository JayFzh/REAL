from ogb.nodeproppred import NodePropPredDataset
import torch
import dgl

# Download the ogbn-products dataset
dataset = NodePropPredDataset(name='ogbn-products', root='./data')
graph, labels = dataset[0]

# Convert to DGLGraph
g = dgl.graph((graph['edge_index'][0], graph['edge_index'][1]))
g.ndata['feat'] = torch.tensor(graph['node_feat'])
g.ndata['label'] = torch.from_numpy(labels).squeeze()

print(g)
print("Node feature shape:", g.ndata['feat'].shape)
print("Label shape:", g.ndata['label'].shape)
