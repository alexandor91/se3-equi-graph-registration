# Copyright (c) Alexander.Kang alexander.kang@tum.de
# Equi-GSPR: Equivariant Graph Model for Sparse Point Cloud Registration
# Please cite the following papers if you use any part of the code.
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
from torch_scatter import scatter_add

torch.cuda.manual_seed(2)

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
        # h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        # h = h.relu()
        
        # h = self.conv3(h=h, pos=pos, edge_index=edge_index)
        # h = h.relu()
        print("&&&&&&&&&&&&")
        print(h.shape)             ##############torch.Size([2048, 64])
        # Global Pooling:
        #h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
        self.to(self.device)
        return h

class SO3TensorProductLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SO3TensorProductLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # MLP to process the tensor product result
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * input_dim, 2 * output_dim),
            nn.ReLU(),
            nn.Linear(2 * output_dim, output_dim)
        )

    def forward(self, x):
        # x: [num_edges, input_dim * input_dim] - the flattened SO(3) matrix for each edge
        num_edges = x.size(0)  # Get the number of edges
        
        # Reshape x back to [num_edges, input_dim, input_dim] before applying batch matrix multiplication
        x_reshaped = x.view(num_edges, self.input_dim, self.input_dim)  # Reshape to [num_edges, 3, 3]

        # Perform the tensor product of the matrix with itself (batch matrix multiplication)
        tensor_product = torch.bmm(x_reshaped, x_reshaped.transpose(1, 2))  # [num_edges, input_dim, input_dim]
        # Flatten the tensor product for MLP processing
        tensor_product_flat = tensor_product.view(num_edges, -1)  # [num_edges, input_dim * input_dim]

        # Apply the MLP to the tensor product result
        return self.mlp(tensor_product_flat)  # [num_edges, output_dim]

# The compute_so3_matrix function for N x 3 (no batch size)
def compute_so3_matrix(x, graph_idx):
    epsilon = 1e-8
    threshold = 1e-6  # Threshold to detect near-zero values
    
    print("@@@@@@@@@so(3)  matrix@@@@@@@@")
    idx_i = graph_idx[0]  # Source node indices
    idx_k = graph_idx[1]  # Target node indices

    x_i = x[idx_i]  # Coordinates of source nodes
    x_k = x[idx_k]  # Coordinates of target nodes

    # Relative coordinates and normalizing
    rel_coord = x_i - x_k
    rel_coord_norm = rel_coord / (rel_coord.norm(dim=1, keepdim=True) + epsilon)

    # Cross product between node coordinates
    cross_prod = torch.cross(x_i, x_k, dim=1)
    cross_prod_norm = cross_prod / (cross_prod.norm(dim=1, keepdim=True) + epsilon)

    # Basis vectors for the SO(3) matrix
    a_ik = rel_coord_norm
    b_ik = cross_prod_norm
    c_ik = torch.cross(rel_coord_norm, cross_prod_norm, dim=1)

    # Check for near-zero or degenerate matrices, and replace with identity if necessary
    a_ik_norm = a_ik.norm(dim=1)
    b_ik_norm = b_ik.norm(dim=1)
    c_ik_norm = c_ik.norm(dim=1)
    
    mask = (a_ik_norm < threshold) | (b_ik_norm < threshold) | (c_ik_norm < threshold)
    
    # Initialize with identity matrix for unstable or degenerate matrices
    so3_matrix = torch.stack([a_ik, b_ik, c_ik], dim=2)
    identity_matrix = torch.eye(3, device=x.device).unsqueeze(0).expand(so3_matrix.shape[0], -1, -1)

    # Apply mask to replace degenerate matrices with the identity matrix
    so3_matrix[mask] = identity_matrix[mask]

    so3_flat = so3_matrix.view(-1, 9)  # Flatten to shape (N, 9)
    
    # Debugging prints (optional)
    # print(a_ik)
    # print(b_ik)
    # print(c_ik)
    # print(so3_matrix)
    return so3_flat

class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer, basic module to build up the EGNN network model.
    Supports batched inputs.
    """
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False, device='cuda:0'):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1
        self.device = device

        # Edge MLP, with 9 additional dimensions for SO(3) flattened feature
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d + 9, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, edge_coords_nf, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=1e-3)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, edge_coords_nf),
                nn.Sigmoid())
        
        # Initialize SO(3) Tensor Product Layer
        self.so3_tensor_product = SO3TensorProductLayer(input_dim=3, output_dim=hidden_nf)

    def _batch_edges(self, edge_index, batch_size, num_nodes):
        """
        Batch edge indices by shifting the node indices per graph in batch.
        """
        batched_edges = []
        for i in range(batch_size):
            batched_edges.append(edge_index + i * num_nodes)
        return torch.cat(batched_edges, dim=1)

    def edge_model(self, source, target, radial, edge_attr, coord, edge_index, batch_size):
        # Compute the SO(3) matrix features (batch_size * N x 9)
        so3_flat = compute_so3_matrix(coord, edge_index)
        
        if edge_attr is None:
            out = torch.cat([source, target, radial, so3_flat], dim=1)
        else:
            # Concatenate source, target, radial, edge_attr, and SO(3) flattened features
            out = torch.cat([source, target, radial, edge_attr, so3_flat], dim=1)

        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, h, edge_index, edge_attr, node_attr, batch_size):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        if node_attr is not None:
            agg = torch.cat([h, agg, node_attr], dim=1)
        else:
            agg = torch.cat([h, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = h + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, batch_size):
        row, col = edge_index
        if edge_feat is not None:
            trans = coord_diff * self.coord_mlp(edge_feat)
        trans = coord_diff

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

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, batch_size=1):
        # Flatten batch dimension for h and coordinates
        h = h.view(-1, h.size(-1))  # Flatten batch for node features
        coord = coord.view(-1, coord.size(-1))  # Flatten batch for coordinates
        num_nodes = h.size(0) // batch_size

        # Batch edge indices by shifting node indices per graph
        batched_edges = self._batch_edges(edge_index, batch_size, num_nodes)

        # Compute radial and coordinate differences
        radial, coord_diff = self.coord2radial(batched_edges, coord)

        # Edge model with SO(3) features concatenated
        edge_feat = self.edge_model(h[batched_edges[0]], h[batched_edges[1]], radial, edge_attr, coord, batched_edges, batch_size)
        
        # Update coordinates
        coord = self.coord_model(coord, batched_edges, coord_diff, edge_feat, batch_size)
        
        # Update node features
        h, agg = self.node_model(h, batched_edges, edge_feat, node_attr, batch_size)

        # Reshape back to batch dimensions
        h = h.view(batch_size, num_nodes, -1)
        coord = coord.view(batch_size, num_nodes, -1)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cuda:0', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=True, normalize=False, tanh=False):
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages
        :param tanh: Whether to apply a tanh activation to the coordinate update
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(n_layers):
            self.add_module(f"gcl_{i}", E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                              act_fn=act_fn, residual=residual, attention=attention,
                                              normalize=normalize, tanh=tanh))

    def forward(self, h, x, edges, edge_attr, batch_size):
        """
        h: Node features tensor of shape [batch_size, num_nodes, node_features]
        x: Node coordinates tensor of shape [batch_size, num_nodes, 3]
        edges: Edge index tensor of shape [2, num_edges] (for each graph)
        edge_attr: Edge attributes tensor of shape [batch_size * num_edges, edge_features]
        """
        batch_size, num_nodes, _ = h.size()
        h = self.embedding_in(h.view(-1, h.size(-1)))  # Flatten batch for embedding
        x = x.view(-1, 3)  # Flatten batch for coordinates

        # Batch edge indices by shifting node indices per graph in batch
        batched_edges = self._batch_edges(edges, batch_size, num_nodes)

        for i in range(self.n_layers):
            h, x, _ = self._modules[f"gcl_{i}"](h, batched_edges, x, edge_attr=edge_attr)

        h = self.embedding_out(h)
        return h.view(batch_size, num_nodes, -1), x.view(batch_size, num_nodes, 3)

    def _batch_edges(self, edges, batch_size, num_nodes):
        """
        Shift the node indices in `edges` for each graph in the batch to handle multiple graphs in the batch.
        """
        edges_shifted = []
        for i in range(batch_size):
            edges_shifted.append(edges + i * num_nodes)
        return torch.cat(edges_shifted, dim=1)

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

def get_edges_from_idx(graph_idx):
    """
    Converts the graph index to a format compatible with edge batching.
    `graph_idx` is expected to be in the shape of [2, num_edges].
    """
    src, dst = graph_idx[0], graph_idx[1]
    return [src, dst]

def get_edges_batch(graph_idx, n_nodes, batch_size):
    """
    Adapted version of the original function to use a sparse edge index `graph_idx`.
    """
    # Get the edges (source and target indices) from graph_idx
    edges = get_edges_from_idx(graph_idx)
    
    # Create edge attributes (dummy values set to 1)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1).cuda()
    
    # Assuming the edges are already on the GPU, remove the redundant torch.LongTensor conversion
    edges = [edges[0], edges[1]]  # These should already be CUDA tensors
    
    # Handle batch size
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            # Adjust the edges by shifting the node indices for each batch
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        
        # Concatenate rows and columns across batches (no need to re-cast to LongTensor or move to GPU)
        edges = [torch.cat(rows), torch.cat(cols)]
    
    return edges, edge_attr
# def get_edges_batch(n_nodes, batch_size):
#     edges = get_edges(n_nodes)
#     edge_attr = torch.ones(len(edges[0]) * batch_size, 1).cuda()
#     edges = [torch.LongTensor(edges[0]).cuda(), torch.LongTensor(edges[1]).cuda()]
#     if batch_size == 1:
#         return edges, edge_attr
#     elif batch_size > 1:
#         rows, cols = [], []
#         for i in range(batch_size):
#             rows.append(edges[0] + n_nodes * i)
#             cols.append(edges[1] + n_nodes * i)
#         edges = [torch.cat(rows).cuda(), torch.cat(cols).cuda()]
#     return edges, edge_attr


if __name__ == "__main__":
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos = torch.rand(2048, 3).to(dev)
    edge = torch.randint(1, 10, (2, 768)).to(dev)
    batch = torch.tensor([1]).to(dev)

#   dataset.transform = T.Compose([SamplePoints(num=256), KNNGraph(k=6)])

#   data = dataset[0]
#   y = [1, 3]
#   Data(pos, edge_index=[2, 1536], y)
    k = 12  ######knn nearest neighbouring for grpah initialization
    graph_idx = knn_graph(pos,k,loop=False)
    index = torch.LongTensor([1, 0])
    new_graph_idx = torch.zeros_like(graph_idx)
    new_graph_idx[index] = graph_idx
    ks_graph_coords = pos[new_graph_idx]
    # edgegraph = KNNGraph(k=12) #RadiusGraph(r=0.05, loop = False, max_num_neighbors = 15, num_workers = 6)
    print("###graph init begins###########")
    print(index.shape)       ##########orch.Size([2])
    print(graph_idx.shape)   #######torch.Size([2, 24576])
    print(pos.shape)     ##########torch.Size([2048, 3])
    print(ks_graph_coords.shape) ######torch.Size([2048, 64])
    # print(pos) 
    # print(new_graph_idx)
    # print(ks_graph)
    feature_encoder = PointNet().to(dev)
    feats = feature_encoder(pos, edge, batch).to(dev)

    print(feats.shape)    ##############torch.Size([2048, 64])
    # Dummy parameters
    n_nodes = torch.tensor([pos.shape[0]]).to(dev)
    n_feat =  torch.tensor([feats.shape[1]]).to(dev)
    x_dim = torch.tensor([pos.shape[1]]).to(dev)

    # Dummy variables h, x and fully connected edges
    h = torch.ones(batch * n_nodes, n_feat).to(dev)
    x = torch.ones(batch * n_nodes, x_dim).to(dev)
    edges, edge_attr = get_edges_batch(graph_idx, n_nodes, batch)

    # Initialize EGNN
    egnn = EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1).cuda()

    print(x.shape)
    print(h.shape)
    print(n_nodes)
    print(x_dim)
    print(batch)
    print(edges[0].shape)
    # Run EGNN
    h, x = egnn(h, x, edges, edge_attr)
