from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv

model = Sequential('x, edge_index', [
    (GCNConv(32, 32), 'x, edge_index -> x'),
    ReLU(inplace=True),
    (GCNConv(32, 32), 'x, edge_index -> x'),
    ReLU(inplace=True),
    Linear(32, 32),
])

total_params = sum(p.numel() for p in model.parameters())
total_size_bytes = total_params * 4
print(total_params)