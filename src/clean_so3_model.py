
import torch
from torch import nn, Tensor
import torch.optim as optim
# from egnn_pytorch import EGNN
# from se3_transformer_pytorch import SE3Transformer
# import egcnModel
from gcnLayer import GraphConvolution, GlobalPooling
#from se3_transformer_pytorch.se3_transformer_pytorch import SE3Transformer
#from se3_transformer_pytorch.irr_repr import rot
#from se3_transformer_pytorch.utils import torch_default_dtype, fourier_encode
import os, errno
import numpy as np
import wandb
import json
#from sklearn.neighbors import NearestNeighbors
#from datasets import TUMDataset

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected
from torch_geometric.nn import global_max_pool, MessagePassing
# from torch_geometric.data import Datao8
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.transforms import SamplePoints, KNNGraph
import torch_geometric.transforms as T
from torch_cluster import knn_graph

torch.cuda.manual_seed(2)

class SO3TensorProductLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SO3TensorProductLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * input_dim, 2 * output_dim),
            nn.ReLU(),
            nn.Linear(2 * output_dim, output_dim)
        )

    def forward(self, x):
        # x: [num_edges, input_dim]
        batch_size = x.size(0)
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.input_dim)
        x_transposed = x.unsqueeze(1).expand(-1, self.input_dim, -1)
        tensor_product = torch.bmm(x_expanded, x_transposed)  # [num_edges, input_dim, input_dim]
        tensor_product_flat = tensor_product.view(batch_size, -1)  # [num_edges, input_dim * input_dim]
        return self.mlp(tensor_product_flat)  # [num_edges, output_dim]

class E_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False, device='cuda:0'):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.device = device

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + 3, hidden_nf),  # +3 for relative position embedding
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1, bias=False)
        )

        self.so3_layer = SO3TensorProductLayer(hidden_nf, hidden_nf)

    def edge_model(self, source, target, radial, edge_attr):
        # source, target: [num_edges, input_nf]
        # radial: [num_edges, 1]
        # edge_attr: [num_edges, edges_in_d] or None
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)  # [num_edges, hidden_nf]
        out = self.so3_layer(out)  # [num_edges, hidden_nf]
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)  # [num_edges, hidden_nf]
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)  # [num_nodes, 3]
        
        # Integrate relative position embedding
        relative_pos_embedding = coord_diff / (torch.norm(coord_diff, dim=1, keepdim=True) + self.epsilon)
        h_with_pos = torch.cat([h, relative_pos_embedding], dim=1)  # [num_nodes, input_nf + 3]
        
        h, agg = self.node_model(h_with_pos, edge_index, edge_feat, node_attr)  # h: [num_nodes, output_nf]

        return h, coord, edge_attr

class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cuda:0', act_fn=nn.SiLU(), n_layers=1, residual=True, attention=False, normalize=False, tanh=False):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))

    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)  # [num_nodes, hidden_nf]
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)  # [num_nodes, out_node_nf]
        return h, x

class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden
        # node dimensionality plus point dimensionality (=3).
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self,
        h_j: Tensor,
        pos_j: Tensor,
        pos_i: Tensor,
    ) -> Tensor:
        # h_j: The features of neighbors as shape [num_edges, in_channels]
        # pos_j: The position of neighbors as shape [num_edges, 3]
        # pos_i: The central node position as shape [num_edges, 3]

        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)

class PointNet(torch.nn.Module):
    def __init__(self, in_num_feature=3, hidden_num_feature=32, output_num_feature=64, device='cuda:0'):
        super().__init__()
        self.hidden_num_feature = hidden_num_feature
        self.conv1 = PointNetLayer(in_num_feature, self.hidden_num_feature)
        self.conv2 = PointNetLayer(self.hidden_num_feature, output_num_feature)
        # self.conv3 = PointNetLayer(2*self.hidden_num_feature, output_num_feature)
        self.device = device
        
    def forward(self,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:

        # Perform two-layers of message passing:
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        # h = self.conv3(h=h, pos=pos, edge_index=edge_index)
        # h = h.relu()
        print("&&&&&&&&&&&&")
        print(h.shape)
        # Global Pooling:
        #h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
        self.to(self.device)
        return h

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges

def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1).cuda()
    edges = [torch.LongTensor(edges[0]).cuda(), torch.LongTensor(edges[1]).cuda()]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows).cuda(), torch.cat(cols).cuda()]
    return edges, edge_attr

# ... (rest of the code remains the same)

if __name__ == "__main__":
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos = torch.rand(2048, 3).to(dev)  # [num_nodes, 3]
    edge = torch.randint(1, 10, (2, 768)).to(dev)  # [2, num_edges]
    batch = torch.tensor([1]).to(dev)

    k = 12
    graph_idx = knn_graph(pos, k, loop=False)  # [2, num_edges]
    index = torch.LongTensor([1, 0])
    new_graph_idx = torch.zeros_like(graph_idx)
    new_graph_idx[index] = graph_idx
    ks_graph_coords = pos[new_graph_idx]  # [2, num_edges, 3]

    feature_encoder = PointNet().to(dev)
    feats = feature_encoder(pos, edge, batch).to(dev)  # [num_nodes, output_num_feature]

    # Dummy parameters
    n_nodes = torch.tensor([pos.shape[0]]).to(dev)
    n_feat = torch.tensor([feats.shape[1]]).to(dev)
    x_dim = torch.tensor([pos.shape[1]]).to(dev)

    # Dummy variables h, x and fully connected edges
    h = torch.ones(batch * n_nodes, n_feat).to(dev)  # [num_nodes, n_feat]
    x = torch.ones(batch * n_nodes, x_dim).to(dev)  # [num_nodes, 3]
    edges, edge_attr = get_edges_batch(n_nodes, batch)  # edges: [2, num_edges], edge_attr: [num_edges, 1]

    # Initialize EGNN
    egnn = EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1).to(dev)

    # Run EGNN
    h, x = egnn(h, x, edges, edge_attr)
    print("Output h shape:", h.shape)  # [num_nodes, 1]
    print("Output x shape:", x.shape)  # [num_nodes, 3]