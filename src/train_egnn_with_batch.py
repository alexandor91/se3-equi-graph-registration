<<<<<<< HEAD

=======
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
# from egnn_pytorch import EGNN
# import egcnModel
from gcnLayer import GraphConvolution, GlobalPooling
#from se3_transformer_pytorch.se3_transformer_pytorch import SE3Transformer
#from se3_transformer_pytorch.irr_repr import rot
#from se3_transformer_pytorch.utils import torch_default_dtype, fourier_encode
import os, errno
import numpy as np
from scipy.spatial.transform import Rotation as R
import wandb
import json
import argparse
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
from torch.utils.tensorboard import SummaryWriter
import sys
import importlib.util

# Get the path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to the Python path
sys.path.insert(0, project_root)

# Now you can import from datasets
from datasets.ThreeDMatch import ThreeDMatchTrainVal, ThreeDMatchTest  # Replace with your actual class name
from datasets.KITTI import KITTItrainVal, KITTItest  # Replace with your actual class name

torch.cuda.manual_seed(2)

<<<<<<< HEAD
class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden
        # node dimensionality plus point dimensionality (=3).
=======

class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='max')  # "max" aggregation
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

<<<<<<< HEAD
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

=======
    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Start propagating messages
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_j: torch.Tensor, pos_j: torch.Tensor, pos_i: torch.Tensor) -> torch.Tensor:
        # Concatenate features with relative positional differences
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)


<<<<<<< HEAD
class PointNet(torch.nn.Module):
    def __init__(self, in_num_feature=3, hidden_num_feature=32, output_num_feature=32, device='cuda:0'):
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
=======
class PointNet(nn.Module):
    def __init__(self, in_num_feature=3, hidden_num_feature=32, output_num_feature=32, device='cuda:0'):
        super().__init__()
        self.hidden_num_feature = hidden_num_feature
        self.conv1 = PointNetLayer(in_num_feature, hidden_num_feature)
        self.conv2 = PointNetLayer(hidden_num_feature, output_num_feature)
        self.device = device

    def forward(self, pos: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Perform two layers of message passing
<<<<<<< HEAD
=======
        print("##############")
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
>>>>>>> ebc341acecde2ce74385314e139fd7ae7e0b8c1c
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
<<<<<<< HEAD
        # h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        # h = h.relu()
        
        # h = self.conv3(h=h, pos=pos, edge_index=edge_index)
        # h = h.relu()
        # Global Pooling:
        #h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
        self.to(self.device)
=======
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
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

<<<<<<< HEAD
=======

>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
# The compute_so3_matrix function for N x 3 (no batch size)
def compute_so3_matrix(x, graph_idx):
    epsilon = 1e-8
    threshold = 1e-6  # Threshold to detect near-zero values
    
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

# Helper: Compute Edge Distance and Other Features
def compute_edge_features(coord, edge_index):
    row, col = edge_index
    rel_coord = coord[row] - coord[col]  # Relative positions
    dist = torch.norm(rel_coord, dim=1, keepdim=True)  # ∥xi - xj∥
    dot_product = (coord[row] * coord[col]).sum(dim=1, keepdim=True)  # xi ⋅ xj
    return rel_coord, dist, dot_product


# Updated Edge Model with Richer Features
class E_GCL(nn.Module):
<<<<<<< HEAD
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, num_heads=4,
=======
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, num_heads=1,
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
                 act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, tanh=False, device='cuda:0'):
        super(E_GCL, self).__init__()
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh
        self.num_heads = num_heads
        self.device = device

        input_edge = input_nf * 2
        edge_coords_nf = 1
        so3_feat_dim = 9
        feature_dim = input_edge + edges_in_d + edge_coords_nf + so3_feat_dim + 2  # Include distance, dot product

        # Multi-Head Edge MLP
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_nf // num_heads),
                act_fn,
                nn.Linear(hidden_nf // num_heads, hidden_nf // num_heads)
            ) for _ in range(num_heads)
        ])
        self.layer_norm = nn.LayerNorm(hidden_nf)  # Layer norm after combining heads

        # Node MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )

        # Coordinate MLP
        layer = nn.Linear(hidden_nf, edge_coords_nf, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=1e-3)

        coord_mlp = [
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer
        ]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

    def edge_model(self, source, target, radial, edge_attr, coord, edge_index):
        rel_coord, dist, dot_product = compute_edge_features(coord, edge_index)

        so3_flat = compute_so3_matrix(coord, edge_index)  # From your code
        # dihedral = compute_dihedral_angles(rel_coord, so3_flat[:, :3])  # Placeholder

        # Combine features
        features = [source, target, radial, dist, dot_product, so3_flat]
        if edge_attr is not None:
            features.append(edge_attr)
        out = torch.cat(features, dim=1)

        # Multi-head MLP
        head_outputs = [mlp(out) for mlp in self.edge_mlps]
        combined = torch.cat(head_outputs, dim=1)

        # Layer normalization
        out = self.layer_norm(combined)
        return out

    def node_model(self, x, edge_index, edge_attr):
        row = edge_index[0]
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        #agg = unsorted_segment_mean(edge_attr, row, num_segments=x.size(0))
        out = torch.cat([x, agg], dim=1)
        out = self.node_mlp(out)
        if self.residual:
            out = x + out
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row = edge_index[0]
        trans = coord_diff * self.coord_mlp(edge_feat)
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        # agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord = coord + agg
        return coord


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, -1).unsqueeze(-1)
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, coord, edge_index)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h = self.node_model(h, edge_index, edge_feat)

        # Return h, coord, and a dummy third value
        return h, coord, edge_attr



class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cuda:0', act_fn=nn.SiLU(), n_layers=5, residual=True, attention=True, normalize=False, tanh=False):
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
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

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
        self.to(device)  # Move entire model to device during initialization

    def forward(self, h, x, edges, edge_attr):
        # self.to(self.device)
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, x


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
    edges_concat = [edges[0], edges[1]]  # These should already be CUDA tensors
    # Handle batch size
    if batch_size == 1:
        return edges_concat, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            # Adjust the edges by shifting the node indices for each batch
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        
        # Concatenate rows and columns across batches (no need to re-cast to LongTensor or move to GPU)
        edges_concat = [torch.cat(rows), torch.cat(cols)]    
    return edges_concat, edge_attr

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
# Define the loss function for pose regression (combining quaternion and translation loss)

def rotation_matrix_to_quaternion(rotation_matrix):
    """
    Convert a batched 3x3 rotation matrix to a quaternion.
    Args:
        rotation_matrix (Tensor): Rotation matrix of shape [B, 3, 3].
    Returns:
        Tensor: Quaternion tensor of shape [B, 4].
    """
    B = rotation_matrix.shape[0]
    quaternions = torch.zeros((B, 4), device=rotation_matrix.device)
    
    m = rotation_matrix
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    
    s = torch.where(t > 0, torch.sqrt(1.0 + t) * 2, torch.zeros_like(t))
    mask = (t > 0)
    quaternions[mask, 0] = 0.25 * s[mask]
    quaternions[mask, 1] = (m[mask, 2, 1] - m[mask, 1, 2]) / s[mask]
    quaternions[mask, 2] = (m[mask, 0, 2] - m[mask, 2, 0]) / s[mask]
    quaternions[mask, 3] = (m[mask, 1, 0] - m[mask, 0, 1]) / s[mask]
    
    idx = ~mask
    # Process other cases for indices with idx
    # Implement similar logic for batched inputs for idx
    
    return quaternions

<<<<<<< HEAD
def quaternion_to_matrix(quaternion):
    """
    Converts a batched quaternion into 3x3 rotation matrices.
    Args:
        quaternion (torch.Tensor): A tensor of shape [B, 4] representing (w, x, y, z).
    Returns:
        torch.Tensor: A tensor of shape [B, 3, 3] representing rotation matrices.
    """
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    
    rot_matrix = torch.stack([
        [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz]
    ], dim=-1).reshape(-1, 3, 3).to(quaternion.device)
    
    return rot_matrix


def quaternion_to_matrix(quaternion):
    """
    Converts a quaternion into a 3x3 rotation matrix.
    
    Args:
        quaternion (torch.Tensor): A tensor of shape (4,) representing (w, x, y, z).
        
    Returns:
        torch.Tensor: A 3x3 rotation matrix.
    """
    # print("$$$$$$$$$$$$")
    # print(quaternion.shape)
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    
    # Compute the individual terms for the rotation matrix
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    
    # Build the rotation matrix
    rot_matrix = torch.tensor([
        [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz]
    ]).to(quaternion.device)
    
    return rot_matrix
=======
def rotation_matrix_to_quaternion_batch(R):
    """
    Convert a batch of rotation matrices to quaternions.

    Inputs:
        R: (B, 3, 3) - Batch of rotation matrices.

    Outputs:
        quaternions: (B, 4) - Batch of quaternions (w, x, y, z), where w is the scalar part.
    """
    B = R.shape[0]
    
    # Allocate output tensor for quaternions
    quaternions = torch.zeros((B, 4), device=R.device, dtype=R.dtype)

    # Extract trace of each rotation matrix
    trace = torch.einsum('bii->b', R)  # Sum of diagonal elements (B,)

    for b in range(B):
        if trace[b] > 0.0:
            # Case: Trace is positive
            S = torch.sqrt(trace[b] + 1.0) * 2.0  # S = 4 * qw
            qw = 0.25 * S
            qx = (R[b, 2, 1] - R[b, 1, 2]) / S
            qy = (R[b, 0, 2] - R[b, 2, 0]) / S
            qz = (R[b, 1, 0] - R[b, 0, 1]) / S
        elif (R[b, 0, 0] > R[b, 1, 1]) and (R[b, 0, 0] > R[b, 2, 2]):
            # Case: R[0, 0] is the largest diagonal element
            S = torch.sqrt(1.0 + R[b, 0, 0] - R[b, 1, 1] - R[b, 2, 2]) * 2.0  # S = 4 * qx
            qw = (R[b, 2, 1] - R[b, 1, 2]) / S
            qx = 0.25 * S
            qy = (R[b, 0, 1] + R[b, 1, 0]) / S
            qz = (R[b, 0, 2] + R[b, 2, 0]) / S
        elif R[b, 1, 1] > R[b, 2, 2]:
            # Case: R[1, 1] is the largest diagonal element
            S = torch.sqrt(1.0 + R[b, 1, 1] - R[b, 0, 0] - R[b, 2, 2]) * 2.0  # S = 4 * qy
            qw = (R[b, 0, 2] - R[b, 2, 0]) / S
            qx = (R[b, 0, 1] + R[b, 1, 0]) / S
            qy = 0.25 * S
            qz = (R[b, 1, 2] + R[b, 2, 1]) / S
        else:
            # Case: R[2, 2] is the largest diagonal element
            S = torch.sqrt(1.0 + R[b, 2, 2] - R[b, 0, 0] - R[b, 1, 1]) * 2.0  # S = 4 * qz
            qw = (R[b, 1, 0] - R[b, 0, 1]) / S
            qx = (R[b, 0, 2] + R[b, 2, 0]) / S
            qy = (R[b, 1, 2] + R[b, 2, 1]) / S
            qz = 0.25 * S

        quaternions[b] = torch.tensor([qw, qx, qy, qz], device=R.device)

    return quaternions

def quaternion_to_matrix(quaternions):
    """
    Convert a batch of quaternions to a batch of rotation matrices.
    
    Args:
        quaternions (torch.Tensor): Tensor of shape (B, 4), where each quaternion is [qx, qy, qz, qw].
        
    Returns:
        torch.Tensor: Tensor of shape (B, 3, 3), batch of rotation matrices.
    """
    # Ensure the input has the correct shape
    assert quaternions.shape[-1] == 4, "Input quaternions should have shape (B, 4)"
    
    # Normalize the quaternions to ensure unit length
    quaternions = F.normalize(quaternions, p=2, dim=-1)
    
    # Extract components of the quaternion
    qx, qy, qz, qw = quaternions.unbind(dim=-1)
    
    # Compute the rotation matrix components
    r00 = 1 - 2 * (qy ** 2 + qz ** 2)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)
    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx ** 2 + qz ** 2)
    r12 = 2 * (qy * qz - qx * qw)
    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx ** 2 + qy ** 2)
    
    # Stack components into the rotation matrix
    rotation_matrix = torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )  # Shape: (B, 3, 3)
    
    return rotation_matrix

>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c

def matrix_log(R):
    """
    Compute the batched matrix logarithm of 3x3 rotation matrices R.
    Args:
        R: Tensor of shape [B, 3, 3].
    Returns:
        Tensor: Matrix logarithms of shape [B, 3, 3].
    """
    trace = R.diagonal(dim1=1, dim2=2).sum(dim=-1)
    theta = torch.acos((trace - 1) / 2)
    
    theta_abs = torch.abs(theta)
    log_R = torch.where(
        theta_abs < 1e-6,
        torch.zeros_like(R),
        theta[:, None, None] / (2 * torch.sin(theta)[:, None, None]) * (R - R.transpose(1, 2))
    )
    
    return log_R


def center_and_normalize(src_pts, tar_pts):
    """
    Normalize batched source and target points by centering them at the origin and scaling them to have unit norm.
    Args:
        src_pts: Tensor of shape [B, N, 3].
        tar_pts: Tensor of shape [B, N, 3].
    Returns:
        Normalized source points [B, N, 3].
        Normalized target points [B, N, 3].
    """
    src_center = src_pts.mean(dim=1, keepdim=True)
    tar_center = tar_pts.mean(dim=1, keepdim=True)
    
    src_pts_centered = src_pts - src_center
    tar_pts_centered = tar_pts - tar_center
    
    src_pts_normalized = src_pts_centered / torch.norm(src_pts_centered, dim=2, keepdim=True)
    tar_pts_normalized = tar_pts_centered / torch.norm(tar_pts_centered, dim=2, keepdim=True)
    
    return src_pts_normalized, tar_pts_normalized


class CrossAttentionPoseRegression(nn.Module):
<<<<<<< HEAD
    def __init__(self, egnn, num_nodes=2048, hidden_nf=32, device='cuda:0'):
=======
    def __init__(self, egnn, num_nodes=2048, hidden_nf=33, device='cuda:0'):
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
        super(CrossAttentionPoseRegression, self).__init__()
        self.egnn = egnn
        self.hidden_nf = hidden_nf
        self.num_nodes = num_nodes
        self.device = device

        # MLP layers for compression
<<<<<<< HEAD
        self.mlp_compress = nn.Sequential(
            nn.Linear(num_nodes, num_nodes // 2),
            nn.ReLU(),
            nn.Linear(num_nodes // 2, num_nodes // 4),
            nn.ReLU(),
            nn.Linear(num_nodes // 4, 128)
        )

        # MLP layers for pose regression
        self.mlp_pose = nn.Sequential(
            nn.Linear(16 * 2 * self.hidden_nf, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
=======
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_nf, hidden_nf),
            nn.ReLU(),
            nn.Linear(hidden_nf, hidden_nf // 2),
            nn.ReLU(),
            nn.Linear(hidden_nf // 2, 1),
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
        )

        self.shared_mlp_decoder = nn.Sequential(
            nn.Linear((32 + 3) * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.shallow_mlp_pose = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
        )

        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        self.mean_pooling = nn.AdaptiveAvgPool1d(1)
<<<<<<< HEAD

=======
        self.bn1 = nn.BatchNorm1d(self.hidden_nf)  # Normalize feature dimension
        self.bn2 = nn.BatchNorm1d((self.hidden_nf+3))  # Normalize feature dimension
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
        # Weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

<<<<<<< HEAD
        self.mlp_compress.apply(init_layer)
        self.mlp_pose.apply(init_layer)
        self.shared_mlp_decoder.apply(init_layer)
        self.shallow_mlp_pose.apply(init_layer)

    def forward(self, h_src, x_src, edges_src, edge_attr_src, h_tgt, x_tgt, edges_tgt, edge_attr_tgt, corr, labels):
        batch_size = h_src.size(0)

        # Normalize features
        h_src_norm = F.normalize(h_src, p=2, dim=-1)
        h_tgt_norm = F.normalize(h_tgt, p=2, dim=-1)

        # Concatenate features with coordinates
        h_src = torch.cat([h_src_norm, x_src], dim=-1)  # Shape: [B, N, 35]
        h_tgt = torch.cat([h_tgt_norm, x_tgt], dim=-1)  # Shape: [B, N, 35]

        # Compute similarity matrix
        large_sim_matrix = torch.matmul(h_src_norm, h_tgt_norm.transpose(-1, -2))  # Shape: [B, N, N]
        large_sim_matrix = F.softmax(large_sim_matrix, dim=-1)

        corr_indices = corr.long()  # Shape: [B, M, 2]
        labels = labels.long()  # Shape: [B, M]

        # Correspondence loss computation
        corr_similarity = torch.zeros_like(large_sim_matrix)
        gt_matrix = torch.zeros_like(large_sim_matrix)

        for b in range(batch_size):
            valid_indices = corr_indices[b, labels[b] == 1]
            gt_matrix[b, valid_indices[:, 0], valid_indices[:, 1]] = 1.0
            corr_similarity[b, valid_indices[:, 0], valid_indices[:, 1]] = \
                large_sim_matrix[b, valid_indices[:, 0], valid_indices[:, 1]]

        corr_loss = F.mse_loss(corr_similarity, gt_matrix)

        # Compress features
        compressed_h_src = self.mlp_compress(h_src.permute(0, 2, 1)).permute(0, 2, 1)
        compressed_h_tgt = self.mlp_compress(h_tgt.permute(0, 2, 1)).permute(0, 2, 1)

        # Normalize compressed features
        compressed_h_src_norm = F.normalize(compressed_h_src, p=2, dim=-1)
        compressed_h_tgt_norm = F.normalize(compressed_h_tgt, p=2, dim=-1)

        # Compute similarity matrix for compressed features
        sim_matrix = torch.matmul(compressed_h_src_norm, compressed_h_tgt_norm.transpose(-1, -2))  # Shape: [B, N, N]
        sim_matrix = F.softmax(sim_matrix, dim=-1)

        # Singular Value Decomposition (SVD)
        u, s, v = torch.svd(sim_matrix)  # u: [B, N, N], s: [B, N], v: [B, N, N]
        top_k_sum = s[:, :16].sum(dim=-1)  # Sum of top-k singular values
        rank_loss = F.mse_loss(top_k_sum, torch.tensor(12.0).to(self.device))

        # Total loss
        total_loss = rank_loss + corr_loss

        # Weighted features
        weighted_h_src = torch.matmul(sim_matrix.transpose(-1, -2), compressed_h_src)  # Shape: [B, N, D]
        weighted_h_tgt = torch.matmul(sim_matrix, compressed_h_tgt)  # Shape: [B, N, D]

        # Flatten combined features
        combined_features = torch.cat([weighted_h_src, weighted_h_tgt], dim=-1)
        combined_features_flat = combined_features.view(batch_size, -1)

        # Regress pose (quaternion + translation)
        pose = self.mlp_pose(combined_features_flat)  # Shape: [B, 7]
        quaternion = F.normalize(pose[:, :4], p=2, dim=-1)  # Normalize quaternion
        translation = pose[:, 4:]

        return quaternion, translation, total_loss, h_src_norm, x_src, h_tgt_norm, x_tgt, labels

    
def compute_losses(quaternion, translation, h_src_norm, x_src, h_tgt_norm, x_tgt, gt_labels):
    """
    Args:
        quaternion: Tensor of shape [batch_size, 4] (predicted quaternion).
=======
        # self.mlp_compress.apply(init_layer)
        self.mlp.apply(init_layer)
        # self.shared_mlp_decoder.apply(init_layer)
        # self.shallow_mlp_pose.apply(init_layer)

    def forward(self, h_src, x_src, edges_src, edge_attr_src, h_tgt, x_tgt, edges_tgt, edge_attr_tgt, corr, labels, gt_pose):
        batch_size, num_points, feature_dim = h_src.shape
        device = h_src.device

        # Initialize lists to store the results
        h_src_list, x_src_list = [], []
        h_tgt_list, x_tgt_list = [], []
        # Loop through each batch
        for i in range(h_src.shape[0]):  # Loop over batch dimension
            # Extract the i-th batch
            h_src_i = h_src[i]  # Shape: [N, 32]
            x_src_i = x_src[i]  # Shape: [N, 3]
            h_tgt_i = h_tgt[i]  # Shape: [N, 32]
            x_tgt_i = x_tgt[i]  # Shape: [N, 3]
            
            # Extract the i-th batch's edges and edge attributes
            edges_src_i = edges_src[i]  # Shape: [2, num_edges]
            edge_attr_src_i = edge_attr_src[i]  # Shape: [num_edges, edge_attr_dim]
            edges_tgt_i = edges_tgt[i]  # Shape: [2, num_edges]
            edge_attr_tgt_i = edge_attr_tgt[i]  # Shape: [num_edges, edge_attr_dim]
            
            edges_src_list = list(edges_src_i)  # Convert to list of tensors
            edges_tgt_list = list(edges_tgt_i)  # Convert to list of tensors

            # Pass through the EGNN model
            h_src_i, x_src_i = self.egnn(h_src_i, x_src_i, edges_src_list, edge_attr_src_i)  # Shape: [N, 32], [N, 3]
            h_tgt_i, x_tgt_i = self.egnn(h_tgt_i, x_tgt_i, edges_tgt_list, edge_attr_tgt_i)  # Shape: [N, 32], [N, 3]
            
            # Append the results to the lists
            h_src_list.append(h_src_i)
            x_src_list.append(x_src_i)
            h_tgt_list.append(h_tgt_i)
            x_tgt_list.append(x_tgt_i)

        # Stack the results back into batch tensors
        h_src = torch.stack(h_src_list, dim=0)  # Shape: [b, N, 32]
        x_src = torch.stack(x_src_list, dim=0)  # Shape: [b, N, 3]
        h_tgt = torch.stack(h_tgt_list, dim=0)  # Shape: [b, N, 32]
        x_tgt = torch.stack(x_tgt_list, dim=0)  # Shape: [b, N, 3]

        total_loss = egnn_equi_loss(h_src, x_src, h_tgt, x_tgt, gt_pose[:, :3, :3], gt_pose[:, :3, -1], labels)

        B, N, _ = x_src.shape
        # Row-wise dot product between h_src and h_tgt (resulting in B x N x 1)
        similarity_scores = torch.sum(h_src * h_tgt, dim=-1, keepdim=True)  # [B, N, 1]
        top_k = 2048
        # Get top-k scores and indices (k=128)
        top_scores, top_indices = torch.topk(similarity_scores.squeeze(-1), k=top_k, dim=-1)  # [B, 128]

        # Gather top-k features and corresponding point coordinates
        batch_indices = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, top_k)
        compressed_h_src = torch.gather(h_src, dim=1, index=top_indices.unsqueeze(-1).expand(-1, -1, feature_dim))  # [B, 128, 32]
        compressed_h_tgt = torch.gather(h_tgt, dim=1, index=top_indices.unsqueeze(-1).expand(-1, -1, feature_dim))  # [B, 128, 32]

        compressed_x_src = torch.gather(x_src, dim=1, index=top_indices.unsqueeze(-1).expand(-1, -1, 3))  # [B, 128, 3]
        compressed_x_tgt = torch.gather(x_tgt, dim=1, index=top_indices.unsqueeze(-1).expand(-1, -1, 3))  # [B, 128, 3]

        compressed_labels = torch.gather(labels, dim=1, index=top_indices)  # [B, 128]
        # Step 1: Mask valid correspondences
        valid_mask = labels.squeeze(-1).bool()  # (B, N)

        # Initialize outputs
        R = torch.zeros(B, 3, 3, device=x_src.device)
        t = torch.zeros(B, 3, device=x_src.device)

        for b in range(B):  # Process each batch
            valid_src_points = x_src[b][valid_mask[b]]
            valid_tgt_points = x_tgt[b][valid_mask[b]]
            valid_src_features = h_src[b][valid_mask[b]]
            valid_tgt_features = h_tgt[b][valid_mask[b]]

            if valid_src_points.shape[0] == 0:  # Skip empty batches
                R[b] = torch.eye(3, device=x_src.device)
                t[b] = torch.zeros(3, device=x_src.device)
                continue

            # Compute feature similarity weights
            # feature_diffs = valid_src_features - valid_tgt_features
            # weights = torch.exp(-torch.norm(feature_diffs, dim=1))  # (N_valid,)
            # weights = weights / weights.sum()  # Normalize weights
            weight_scores = torch.sum(valid_src_features * valid_tgt_features, dim=-1)
            weights = torch.nn.functional.softmax(weight_scores, dim=-1)
            
            # Debug check
            if torch.isnan(weights).any() or torch.isinf(weights).any():
                print(f"NaN/Inf in weights for batch {b}: {weights}")
            
            weights = weights / (weights.sum() + 1e-6)  # Fix division instability

            src_centroid = (weights[:, None] * valid_src_points).sum(dim=0, keepdim=True)
            tgt_centroid = (weights[:, None] * valid_tgt_points).sum(dim=0, keepdim=True)

            src_centered = valid_src_points - src_centroid
            tgt_centered = valid_tgt_points - tgt_centroid

            H = (weights[:, None, None] * src_centered[:, :, None] @ tgt_centered[:, None, :]).sum(dim=0)

            # Debug check for NaN/Inf
            if torch.isnan(H).any() or torch.isinf(H).any():
                print(f"NaN/Inf detected in H for batch {b}: {H}")
            
            H += 1e-6 * torch.eye(3, device=H.device)  # Regularization

            # Perform SVD
            U, S, Vt = torch.linalg.svd(H)

            # Clone S to prevent in-place modification issues
            S = S.clone()

            R_b = Vt.T @ U.T

            # Ensure valid rotation matrix
            if torch.det(R_b) < 0:
                Vt[-1, :] *= -1
                R_b = Vt.T @ U.T

            # Compute translation
            t_b = tgt_centroid.squeeze() - R_b @ src_centroid.squeeze()

            # Store results
            R[b] = R_b
            t[b] = t_b
        # Concatenate along the feature dimension
        concat_h = torch.cat([compressed_h_src, compressed_h_tgt], dim=-1)  # Shape: (batch_size, N, 2 * feature_dim)

        # Reshape for MLP
        batch_size, N, hidden_feat_size = concat_h.shape
        concat_h = concat_h.view(-1, hidden_feat_size)  # Reshape to (batch_size * N, 2 * feature_dim)

        # Pass through MLP
        scores = self.mlp(concat_h)  # Shape: (batch_size * N, 1)

        # Reshape back to (batch_size, N, 1)
        scores = scores.view(batch_size, N)

        criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
        corr_loss = criterion(scores, compressed_labels)
        # corr_loss = F.mse_loss(scores, compressed_labels.float())  # Enforce similarity for correct matches

        # # Assemble refined pose
        refined_pose = torch.eye(4, device=x_src.device).repeat(B, 1, 1)  # [B, 4, 4]
        refined_pose[:, :3, :3] = R
        refined_pose[:, :3, 3] = t
        quaternion = F.normalize(rotation_matrix_to_quaternion_batch(refined_pose[:, :4]), p=2, dim=-1)
        translation = refined_pose[:, :3, 3]
        # translation = t
        # Optionally, compute loss
        # svd_loss = self.compute_svd_loss(compressed_h_src, compressed_h_tgt)
        # Combine all losses (if needed)

        return R, t, corr_loss, total_loss, h_src, x_src, h_tgt, x_tgt, labels

    
def compute_losses(rot, translation, h_src_norm, x_src, h_tgt_norm, x_tgt, gt_labels):
    """
    Args:
        rot: Tensor of shape [batch_size, 3, 3] (predicted quaternion).
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
        translation: Tensor of shape [batch_size, 3] (predicted translation).
        h_src_norm: Tensor of shape [batch_size, N, 32] (source point features).
        x_src: Tensor of shape [batch_size, N, 3] (source point coordinates).
        h_tgt_norm: Tensor of shape [batch_size, N, 32] (target point features).
        x_tgt: Tensor of shape [batch_size, N, 3] (target point coordinates).
        gt_labels: Tensor of shape [batch_size, N] (ground-truth labels, 1 for correspondence, 0 otherwise).

    Returns:
        point_error: Mean L2 norm error of predicted vs target points.
        feature_loss: Mean feature similarity loss.
    """
    device = x_src.device
<<<<<<< HEAD
    quaternion = F.normalize(quaternion, p=2, dim=-1)  # Normalize quaternion

    batch_size = quaternion.shape[0]

    # Convert quaternion to rotation matrices
    rotation_matrices = torch.stack([
        quaternion_to_matrix(quaternion[b]).to(device)
        for b in range(batch_size)
    ], dim=0)  # Shape: [batch_size, 3, 3]
=======
    # quaternion = F.normalize(quaternion, p=2, dim=-1)  # Normalize quaternion

    batch_size = rot.shape[0]

    # rotation_matrices = quaternion_to_matrix(quaternion)
    rotation_matrices = rot    
    # torch.stack([
    #     quaternion_to_matrix(quaternion[b]).to(device)
    #     for b in range(batch_size)
    # ], dim=0)  # Shape: [batch_size, 3, 3]
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c

    # Transform source points
    x_src_transformed = torch.matmul(rotation_matrices, x_src.transpose(1, 2)).transpose(1, 2) + translation.unsqueeze(1)

    # Compute point distances
    point_distances = torch.norm(x_src_transformed - x_tgt, dim=-1)  # Shape: [batch_size, N]

    # Apply mask for valid correspondences
<<<<<<< HEAD
    valid_distances = point_distances[gt_labels == 1]
    point_error = torch.mean(valid_distances) if valid_distances.numel() > 0 else torch.tensor(0.0, device=device)

    # Apply LayerNorm for feature stability
    layer_norm = nn.LayerNorm(h_src_norm.shape[-1]).to(device)  # Assumes last dim is feature size
    h_src_norm = layer_norm(h_src_norm)
    h_tgt_norm = layer_norm(h_tgt_norm)
=======
    valid_distances = point_distances * gt_labels  # Elementwise multiplication

    # Calculate the sum of valid distances and the count of valid points for each batch
    sum_valid_distances = torch.sum(valid_distances, dim=1)  # Sum over points for each batch
    num_valid_points = torch.sum(gt_labels, dim=1)           # Count valid points for each batch

    # Avoid division by zero by ensuring num_valid_points is non-zero
    num_valid_points = torch.clamp(num_valid_points, min=1)  # Replace 0 with 1 to avoid division by zero

    # Calculate mean distance for each batch
    batch_mean_distances = sum_valid_distances / num_valid_points  # [batch_size]
    point_error = torch.mean(batch_mean_distances)  # Scalar
    # Apply LayerNorm for feature stability
    # layer_norm = nn.LayerNorm(h_src_norm.shape[-1]).to(device)  # Assumes last dim is feature size
    # h_src_norm = layer_norm(h_src_norm)
    # h_tgt_norm = layer_norm(h_tgt_norm)
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c

    # Compute feature loss
    feature_loss = torch.mean(
        torch.norm(
            h_src_norm[gt_labels == 1] - h_tgt_norm[gt_labels == 1],
            dim=-1
        )
    )

    return point_error, feature_loss

<<<<<<< HEAD
def pose_loss(pred_quaternion, pred_translation, gt_pose, delta=1.5):
    """
    Compute the loss between the predicted pose (quaternion + translation) and the ground truth pose matrix.
    
    Arguments:
        pred_quaternion: Tensor of shape [batch_size, 4] (predicted quaternion).
        pred_translation: Tensor of shape [batch_size, 3] (predicted translation).
        gt_pose: Tensor of shape [batch_size, 4, 4] (ground truth poses).

    Returns:
        rotation_loss: SO(3) geodesic rotation loss.
        translation_loss: Huber loss for translation.
    """
    batch_size = pred_quaternion.shape[0]

    # Extract ground truth rotation and translation
    gt_translation = gt_pose[:, :3, 3]
    gt_rotation = gt_pose[:, :3, :3]

    # Convert ground truth rotation matrices to quaternions
    gt_quaternions = torch.stack([
        rotation_matrix_to_quaternion(gt_rotation[b])
        for b in range(batch_size)
    ], dim=0)

    # Normalize quaternions
    pred_quaternion = F.normalize(pred_quaternion, p=2, dim=-1)
    gt_quaternions = F.normalize(gt_quaternions, p=2, dim=-1)

    # Convert predicted quaternion to rotation matrices
    pred_rotation_matrices = torch.stack([
        quaternion_to_matrix(pred_quaternion[b])
        for b in range(batch_size)
    ], dim=0)

    # Compute geodesic rotation loss
    rotation_loss = torch.mean(
        torch.acos(
            torch.clamp(
                (torch.trace(torch.matmul(pred_rotation_matrices, gt_rotation.transpose(1, 2))) - 1) / 2,
                min=-1,
                max=1
            )
        )
    )

    # Compute translation loss using Huber loss
    huber_loss = nn.HuberLoss(delta=delta)
    translation_loss = huber_loss(pred_translation, gt_translation)

    return rotation_loss, translation_loss

# Function to train for one epoch
def train_one_epoch(model, dataloader, optimizer, device, epoch, writer, use_pointnet=False, log_interval=1, beta=1):
    model.train()
    running_loss = 0.0
    running_corr_loss = 0.0  # Track correspondence rank loss
    running_rot_loss = 0.0  # Track pose rot loss
    running_trans_loss = 0.0  # Track pose trans loss
    running_point_error = 0.0
    running_feature_consensus_error = 0.0

    for batch_idx, (corr, labels, src_pts, tar_pts, src_features, tgt_features, gt_pose) in enumerate(dataloader):
        # Check if any returned data is None, and skip if so
        if corr is None or labels is None or src_pts is None or tar_pts is None or src_features is None or tgt_features is None or gt_pose is None:
            print(f"Skipping batch {batch_idx} due to missing data.")
            continue  # Skip this batch

        # Move the data to the specified device (GPU/CPU)
        corr = corr.to(device)
        labels = labels.to(device)
        xyz_0, xyz_1 = src_pts.to(device), tar_pts.to(device)
        xyz_0_center = src_pts.mean(dim=1, keepdim=True).to(device)
        xyz_1_center = tar_pts.mean(dim=1, keepdim=True).to(device)
        xyz_0 = xyz_0 #- xyz_0_center
        xyz_1 = xyz_1 #- xyz_1_center
        scale1 = xyz_0.norm(dim=2).max(dim=1, keepdim=True).values.to(device)
        scale2 = xyz_1.norm(dim=2).max(dim=1, keepdim=True).values.to(device)
        # xyz_0 = xyz_0 / scale1
        # xyz_1 = xyz_1/ scale2
        feat_0, feat_1 = src_features.to(device), tgt_features.to(device)
        gt_pose = gt_pose.to(device)
        R = gt_pose[:, :3, :3].to(device)
        T = gt_pose[:, :3, 3].to(device)
        # Adjust xyz_0_center to match R's expected shape
        ########### input points shift center #################
        xyz_0_center = xyz_0_center.squeeze(1).unsqueeze(-1)  # From [1, 1, 3] -> [1, 3] -> [1, 3, 1]

        # Ensure xyz_1_center is of shape [1, 3, 1]
        xyz_1_center = xyz_1_center.squeeze(1).unsqueeze(-1)  # From [1, 1, 3] -> [1, 3] -> [1, 3, 1]

        # Ensure T is of shape [1, 3, 1]
        T = T.unsqueeze(-1)  # From [1, 3] -> [1, 3, 1]

        # # Perform the pose norm computation
        # T_norm = (T - xyz_1_center + torch.bmm(R, xyz_0_center)).to(device)  # Result shape: [1, 3, 1]
        # gt_pose[:, :3, :3] = R
        # gt_pose[:, :3, 3] = T_norm.transpose(1, 2)

        # T = T.squeeze(-1)  # Shape: [B, 3] -> T is already [B, 3, 1], no need for transpose.
        # T = T.transpose(1, 2) 
        # print(xyz_0_center.shape)
        # print(gt_pose.shape)
        # # gt_pose_norm = (gt_pose - xyz_1_center + R @ xyz_0_center.T) /scale2
        # gt_pose_norm = (T - xyz_1_center + torch.bmm(R, xyz_0_center.T)).to(device)
        # gt_pose = gt_pose_norm

        # # Step 1: Append a homogeneous coordinate (1) to xyz_0
        # ones = torch.ones(1, xyz_0.size(1), 1, device=xyz_0.device)  # Shape: 1 x N x 1
        # xyz_homo = torch.cat([xyz_0, ones], dim=-1)  # Shape: 1 x N x 4

        # # Step 2: Apply the extrinsic transformation
        # xyz_transformed_homo = torch.matmul(xyz_homo, gt_pose.T)  # Shape: 1 x N x 4

        # # Step 3: Convert back to 3D by dropping the homogeneous coordinate
        # xyz_transformed = xyz_transformed_homo[..., :3]  # Shape: 1 x N x 3
        # xyz_diff = xyz_transformed - xyz_1
        # print("$$$ transformed gt errors $$$$")
        # print(xyz_diff)
        # Initialize KNN graphs for source and target point clouds
        k = 16
        ####### remove the batch size when it is one ##############
        # Squeeze the first dimension if it's 1 (e.g., torch.Size([1, 2048, 3]) -> torch.Size([2048, 3]))
        if xyz_0.dim() == 3 and xyz_0.size(0) == 1:
            xyz_0 = xyz_0.squeeze(0)
        if xyz_1.dim() == 3 and xyz_1.size(0) == 1:
            xyz_1 = xyz_1.squeeze(0)
        if gt_pose.dim() == 3 and gt_pose.size(0) == 1:
            gt_pose = gt_pose.squeeze(0)
        if corr.dim() == 3 and corr.size(0) == 1:
            corr = corr.squeeze(0)
        if labels.dim() == 3 and labels.size(0) == 1:
            labels = labels.squeeze(0)

        graph_idx_0 = knn_graph(xyz_0, k=k, loop=False)
        graph_idx_1 = knn_graph(xyz_1, k=k, loop=False)


        # Descriptor generation using PointNet or directly using feat_0/feat_1
        if use_pointnet:
            feature_encoder = PointNet().to(device)
            feat_0 = feature_encoder(xyz_0, graph_idx_0, None)  # Descriptor for source
            feat_1 = feature_encoder(xyz_1, graph_idx_1, None)  # Descriptor for target
        # else:
        #     feat_0 = feat_0  # Use pre-computed features directly
        #     feat_1 = feat_1  # Use pre-computed features directly

        if feat_0.dim() == 3 and feat_0.size(0) == 1:
            feat_0 = feat_0.squeeze(0)
        if feat_1.dim() == 3 and feat_1.size(0) == 1:
            feat_1 = feat_1.squeeze(0)

        # feats_0 = feature_encoder(xyz_0, graph_idx_0, None)  # Descriptor for source
        # feats_1 = feature_encoder(xyz_1, graph_idx_1, None)  # Descriptor for target
        # Initialize edges and edge attributes for source and target
        edges_0, edge_attr_0 = get_edges_batch(graph_idx_0, xyz_0.size(0), 1)
        edges_1, edge_attr_1 = get_edges_batch(graph_idx_1, xyz_1.size(0), 1)

        # Zero the gradients
        optimizer.zero_grad()

        # xyz_0 = xyz_0 - xyz_0_center.squeeze(-1)
        # xyz_1 = xyz_1 - xyz_1_center.squeeze(-1)
    
        # # Forward pass through the model
        quaternion, translation, corr_loss, h_src_norm, x_src, h_tgt_norm, x_tgt, gt_labels = model(feat_0, xyz_0, edges_0, edge_attr_0, feat_1, xyz_1, edges_1, edge_attr_1, corr, labels)
        #corr_loss
        point_error, feature_loss = compute_losses(quaternion, translation, h_src_norm, x_src, h_tgt_norm, x_tgt, gt_labels)
        # # Compute pose loss
        rot_loss, trans_loss = pose_loss(quaternion, translation, gt_pose, delta=1.5)

        # # Combine pose and correspondence loss
        loss = rot_loss + trans_loss + beta*corr_loss 

        # # Backward pass and optimization step
        loss.backward()
        ########### clip gradients #################
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        running_rot_loss += rot_loss.item()
        running_trans_loss += trans_loss.item()
        running_corr_loss += corr_loss.item()
        running_point_error += point_error.item()
        running_feature_consensus_error += feature_loss.item()
        # print("$$$$ iteration runing loss $$$$$")
        # print(running_loss)
        # Print loss for every log_interval batches
        if (batch_idx+1) % log_interval == 0:
            print(f'Iteration {batch_idx}/{len(train_loader)} - Loss: {loss.item():.6f}')

            # Log the average loss for this epoch
        # writer.add_scalar('Loss/train', avg_loss, epoch)

    # Return the average losses over the entire train dataset
    avg_loss = running_loss / len(dataloader)
    avg_rot_loss = running_rot_loss / len(dataloader)
    avg_trans_loss = running_trans_loss / len(dataloader)
    avg_corr_loss = running_corr_loss / len(dataloader)
    avg_point_error = running_point_error / len(dataloader)
    avg_feature_consensus_error = running_feature_consensus_error / len(dataloader)

    print(f'Training Loss: {avg_loss:.6f} | Pose rot Loss: {avg_rot_loss:.6f} | Pose trans Loss: {avg_trans_loss:.6f} | Point error: {avg_point_error:.6f} | Corr loss: {corr_loss} | Feature error:  {avg_feature_consensus_error:.6f}')

=======
def egnn_equi_loss(h_src, x_src, h_tgt, x_tgt, R_gt, t_gt, labels):
    """
    Compute EGNN-based equivariant loss with:
    1. Rotation consistency loss
    2. Feature similarity loss
    
    Args:
        h_src: (batch_size, N, 32) - Source node features
        x_src: (batch_size, N, 3) - Source node positions
        h_tgt: (batch_size, N, 32) - Target node features
        x_tgt: (batch_size, N, 3) - Target node positions
        R_gt: (batch_size, 3, 3) - Ground-truth rotation
        t_gt: (batch_size, 3) - Ground-truth translation
        labels: (batch_size, N) - Binary correspondence labels (1 = correct, 0 = incorrect)
    
    Returns:
        Total loss (scalar tensor)
    """
    batch_size, N, _ = x_src.shape

    ### 🔹 Step 1: Rotation Consistency Loss
    x_src_transformed = torch.einsum("bij,bnj->bni", R_gt, x_src) + t_gt[:, None, :]
    chamfer_loss = F.mse_loss(x_src_transformed, x_tgt, reduction="none")  # (batch_size, N, 3)
    chamfer_loss = chamfer_loss.sum(dim=-1)  # (batch_size, N)
    rotation_loss = (chamfer_loss * labels).mean()  # Only penalize correct correspondences

    ### 🔹 Step 2: Feature Similarity Loss (Equivariance)
    feature_similarity = F.cosine_similarity(h_src, h_tgt, dim=-1)  # (batch_size, N)
    feature_loss = F.mse_loss(feature_similarity, labels.float())  # Enforce similarity for correct matches

    ### 🔹 Total Loss
    total_loss = rotation_loss + feature_loss  # Combine both losses

    return total_loss


def pose_loss(pred_rot, pred_translation, gt_pose, delta=1.5):
    """
    Compute the loss between the predicted pose (rot + translation) and the ground truth pose matrix.
    
    Arguments:
    - pred_rot: Predicted rot of shape [B, 3, 3]
    - pred_translation: Predicted translation of shape [B, 3]
    - gt_pose: Ground truth pose as a batch of 4x4 matrices, shape [B, 4, 4]
    - delta: Huber loss delta parameter
    
    Returns:
    - Rotation loss: Tensor of shape [B]
    - Translation loss: Tensor of shape [B]
    """
    batch_size = pred_rot.size(0)
    
    # Extract ground truth translation and rotation
    gt_translation = gt_pose[:, :3, 3]  # Shape: [B, 3]
    gt_rotation = gt_pose[:, :3, :3]    # Shape: [B, 3, 3]
    # gt_quaternion = rotation_matrix_to_quaternion_batch(gt_rotation)  # Convert [B, 3, 3] to [B, 4]

    # Normalize ground truth and predicted quaternions
    # gt_quaternion = F.normalize(gt_quaternion, p=2, dim=-1)      # Shape: [B, 4]
    # pred_quaternion = F.normalize(pred_quaternion, p=2, dim=-1)  # Shape: [B, 4]

    # # Convert predicted quaternion to rotation matrix
    # pred_rotation = quaternion_to_matrix(pred_quaternion)  # Shape: [B, 3, 3]

    # pred_rotation_3x3 = R[:, :3, :3]
    # gt_rotation_3x3 = gt_rotation[:, :3, :3]

    # Compute rotation difference R_diff = R_est^T * R_gt
    R_diff = torch.bmm(pred_rot.transpose(1, 2), gt_rotation)  # Batch matrix multiplication
    
    # Convert rotation matrices to axis-angle representation
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))  # Ensure numerical stability
    rotvec = torch.zeros_like(R_diff[:, 0])
    mask = theta > 1e-6  # Avoid division by zero

    if mask.any():
        r = torch.stack([R_diff[:, 2, 1] - R_diff[:, 1, 2],
                            R_diff[:, 0, 2] - R_diff[:, 2, 0],
                            R_diff[:, 1, 0] - R_diff[:, 0, 1]], dim=1) / (2 * torch.sin(theta).unsqueeze(1))
        rotvec[mask] = r[mask] * theta[mask].unsqueeze(1)
    
    # Compute rotation error in degrees
    rot_error = torch.rad2deg(torch.norm(rotvec, dim=1))

    # Compute translation error (L2 norm)
    trans_error = torch.norm(pred_translation - gt_translation, dim=1)

    pred_rotation_T = pred_rot.transpose(-1, -2)
    R = torch.matmul(pred_rotation_T, gt_rotation)
    trace_R = R.diagonal(dim1=-2, dim2=-1).sum(-1)

    rotation_loss = torch.arccos(torch.clamp((trace_R - 1) / 2, min=-1, max=1))

    # Translation loss (Cosine similarity-based)
    dot_product = torch.sum(pred_translation * gt_translation, dim=-1)  # Shape: [B]
    norm_pred = torch.norm(pred_translation, dim=-1)                    # Shape: [B]
    norm_gt = torch.norm(gt_translation, dim=-1)                        # Shape: [B]
    cosine_similarity = dot_product / (norm_pred * norm_gt)             # Shape: [B]

    translation_loss = torch.arccos(torch.clamp(cosine_similarity, min=-1, max=1))  # Shape: [B]

    return rotation_loss, translation_loss

def train_one_epoch(model, dataloader, optimizer, device, epoch, writer, use_pointnet, log_interval, beta):
    model.train()
    total_loss = 0.0

    for batch_idx, (corr, labels, src_pts, tar_pts, src_features, tgt_features, gt_pose) in enumerate(dataloader):
        if corr is None or labels is None or src_pts is None or tar_pts is None or src_features is None or tgt_features is None or gt_pose is None:
            continue

        optimizer.zero_grad()

        # Move data to the device
        corr = corr.to(device)
        labels = labels.to(device)
        xyz_0, xyz_1 = src_pts.to(device), tar_pts.to(device)
        feat_0, feat_1 = src_features.to(device), tgt_features.to(device)
        gt_pose = gt_pose.to(device)

        # Flatten the input points for KNN computation
        # xyz_0_flat = xyz_0.view(-1, xyz_0.size(-1))  # Shape: [batch_size * num_points, 3]
        # xyz_1_flat = xyz_1.view(-1, xyz_1.size(-1))

        # Input: xyz_0 and xyz_1 are tensors of shape [batch_size, 2048, 3]
        batch_size, num_points, _ = xyz_0.shape

        # Generate batch indices (ensures k-NN only considers points within the same batch)
        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(num_points)  # Shape: [batch_size * num_points]

        # Flatten input points for k-NN computation
        xyz_0_flat = xyz_0.view(-1, 3)  # Shape: [batch_size * num_points, 3]
        xyz_1_flat = xyz_1.view(-1, 3)  # Shape: [batch_size * num_points, 3]

        # Compute k-NN graphs (ensuring edges are within batch instances)
        k = 16
        # graph_idx_0 = knn_graph(xyz_0_i, k=k, loop=False)
        # graph_idx_1 = knn_graph(xyz_1_i, k=k, loop=False)

        graph_idx_0_list = []
        graph_idx_1_list = []

        for i in range(batch_size):
            # Compute kNN graph for each batch
            graph_idx_0_i = knn_graph(xyz_0[i], k=k, loop=True)  # Shape: (2, num_edges)
            graph_idx_1_i = knn_graph(xyz_1[i], k=k, loop=True)  # Shape: (2, num_edges)

            graph_idx_0_list.append(graph_idx_0_i)  # Store edge indices
            graph_idx_1_list.append(graph_idx_1_i)

        # Stack to form (batch_size, 2, num_edges)
        graph_idx_0 = torch.stack(graph_idx_0_list, dim=0)  # Shape: (batch_size, 2, num_edges)
        graph_idx_1 = torch.stack(graph_idx_1_list, dim=0)  # Shape: (batch_size, 2, num_edges)

        # # Debugging: Check graph indices
        # print(f"graph_idx_0 shape: {graph_idx_0.shape}")  # Expected: (2, batch_size * num_points * k)
        # print(f"graph_idx_1 shape: {graph_idx_1.shape}")  # Expected: (2, batch_size * num_points * k)

        # Ensure edges are contained within each batch
        src, dst = graph_idx_0[0], graph_idx_0[1]
        src_batch = batch_indices[src]  # Batch indices for source nodes
        dst_batch = batch_indices[dst]  # Batch indices for target nodes
        cross_batch_edges = (src_batch != dst_batch).sum().item()
        print(f"[DEBUG] Cross-batch edges (should be 0): {cross_batch_edges}")

        # If there are cross-batch edges, raise an error
        if cross_batch_edges > 0:
            raise ValueError("Cross-batch edges detected. Ensure `batch_indices` is correct and `knn_graph` respects batch boundaries.")


        # If using PointNet, encode the features
        if use_pointnet:
            # Use the updated PointNet model
            pointnet = PointNet(in_num_feature=3, hidden_num_feature=32, output_num_feature=32, device=xyz_0.device).to(device)

            # Compute features
            feat_0 = pointnet(xyz_0_flat, graph_idx_0, batch_indices).view(batch_size, num_points, -1)  # [batch_size, 2048, 32]
            feat_1 = pointnet(xyz_1_flat, graph_idx_1, batch_indices).view(batch_size, num_points, -1)  # [batch_size, 2048, 32]

            # feature_encoder = PointNet().to(device)
            # feat_0 = feature_encoder(xyz_0, graph_idx_0, None)
            # feat_1 = feature_encoder(xyz_1, graph_idx_1, None)
        # Reshape to batch-first format
        graph_idx_0 = graph_idx_0.view(batch_size, 2, k*num_points) #.transpose(0, 1)  # Shape: (batch_size, 2, num_edges_per_batch)
        graph_idx_1 = graph_idx_1.view(batch_size, 2, k*num_points) #.transpose(0, 1)  # Shape: (batch_size, 2, num_edges_per_batch)

        edges_0 = None
        edge_attr_0 = None
        edges_1 = None
        edge_attr_1 = None

        # Initialize lists to store the results
        edges_0_list, edge_attr_0_list = [], []
        edges_1_list, edge_attr_1_list = [], []

        # Loop through each batch
        batch_size = xyz_0.size(0)  # Get batch size

        for i in range(batch_size):
            # Extract the i-th batch
            graph_idx_0_i = graph_idx_0[i]  # Shape: [2, 32768]
            graph_idx_1_i = graph_idx_1[i]  # Shape: [2, 32768]
            xyz_0_i = xyz_0[i]  # Shape: [2048, 3]
            xyz_1_i = xyz_1[i]  # Shape: [2048, 3]
            
            # # Debug prints
            # print(f"Batch {i}:")
            # print(f"graph_idx_0_i shape: {graph_idx_0_i.shape}")
            # print(f"xyz_0_i shape: {xyz_0_i.shape}")
            
            # Call get_edges_batch for the i-th batch
            edges_0_i, edge_attr_0_i = get_edges_batch(graph_idx_0_i, xyz_0_i.size(0), 1)
            edges_1_i, edge_attr_1_i = get_edges_batch(graph_idx_1_i, xyz_1_i.size(0), 1)

            # Stack edges correctly: Convert list [row_tensor, col_tensor] → single tensor of shape (2, N)
            edges_0_i = torch.stack(edges_0_i, dim=0)  # Shape: (2, N)
            edges_1_i = torch.stack(edges_1_i, dim=0)  # Shape: (2, N)

            # Store in lists
            edges_0_list.append(edges_0_i)  # Shape: (2, N)
            edge_attr_0_list.append(edge_attr_0_i)  # Shape: (N, 1)
            edges_1_list.append(edges_1_i)  # Shape: (2, N)
            edge_attr_1_list.append(edge_attr_1_i)  # Shape: (N, 1)

        # Stack across the batch dimension
        edges_0 = torch.stack(edges_0_list, dim=0)  # Shape: (batch_size, 2, N)
        edge_attr_0 = torch.stack(edge_attr_0_list, dim=0)  # Shape: (batch_size, N, 1)
        edges_1 = torch.stack(edges_1_list, dim=0)  # Shape: (batch_size, 2, N)
        edge_attr_1 = torch.stack(edge_attr_1_list, dim=0)  # Shape: (batch_size, N, 1)

        # Forward pass through the model
        rot_mat, translation, corr_loss, ssim_loss, h_src_norm, x_src, h_tgt_norm, x_tgt, gt_labels = model(feat_0, xyz_0, edges_0, edge_attr_0, feat_1, xyz_1, edges_1, edge_attr_1, corr, labels, gt_pose)

        point_errors, feature_loss = compute_losses(rot_mat, translation, h_src_norm, x_src, h_tgt_norm, x_tgt, gt_labels)
        
        # Compute pose loss
        rot_losses, trans_losses = pose_loss(rot_mat, translation, gt_pose, delta=1.5)

        # std_loss = rot_losses.std(unbiased=False) + 1e-6  # Add small value to avoid division by zero

        # # Optionally scale normalized losses
        # scaled_losses = normalized_batch_losses * 1.0  # Scaling factor if desired

        # mean_trans_loss = trans_losses.mean()
        # std_trans_loss = trans_losses.std(unbiased=False) + 1e-6  # Add small value to avoid division by zero
        # normalized_batch_trans_losses = (trans_losses - mean_trans_loss) / std_trans_loss

        rot_loss_mean = rot_losses.mean()  # Normalize rotation loss across the batch
        trans_loss_mean = trans_losses.mean()  # Normalize translation loss across the batch
        corr_loss_mean = corr_loss.mean()
        ssim_loss_mean = ssim_loss.mean()
        point_error = point_errors.mean()
        print("Avg rot, trans, point errors below!")
        print(rot_loss_mean)
        print(trans_loss_mean)
        print(point_error)
        print(ssim_loss_mean)
        # Combine the normalized losses if needed
        total_loss = corr_loss_mean + ssim_loss_mean

        # Combine pose and correspondence losses
        loss = total_loss #+ beta * corr_loss
        torch.autograd.set_detect_anomaly(True)
        
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()

        if batch_idx % log_interval == 0:
            print(f'Epoch [{epoch + 1}], Batch [{batch_idx + 1}], Loss: {loss.item():.6f}')

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}], Average Loss: {avg_loss:.6f}')
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
    return avg_loss


# Function to validate the model on the validation set
<<<<<<< HEAD
def validate(model, dataloader, device, epoch, writer, use_pointnet=False):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    running_corr_loss = 0.0  # Track correspondence rank loss
    running_rot_loss = 0.0  # Track pose rot loss
    running_trans_loss = 0.0  # Track pose trans loss
    running_point_error = 0.0
    running_feature_consensus_error = 0.0

    with torch.no_grad():  # No need to track gradients during validation
        for batch_idx, (corr, labels, src_pts, tar_pts, src_features, tgt_features, gt_pose) in enumerate(dataloader):
            # Check if any returned data is None, and skip if so
            if corr is None or labels is None or src_pts is None or tar_pts is None or src_features is None or tgt_features is None or gt_pose is None:
                print(f"Skipping batch {batch_idx} due to missing data.")
                continue  # Skip this batch

            # Move the data to the specified device (GPU/CPU)
=======
def validate(model, dataloader, device, epoch, writer, use_pointnet=False, beta=1):
    model.eval()
    running_loss = 0.0
    running_corr_loss = 0.0
    running_rot_loss = 0.0
    running_trans_loss = 0.0
    running_point_error = 0.0
    running_feat_loss = 0.0
    running_feature_consensus_error = 0.0

    with torch.no_grad():
        for batch_idx, (corr, labels, src_pts, tar_pts, src_features, tgt_features, gt_pose) in enumerate(dataloader):
            if any(x is None for x in [corr, labels, src_pts, tar_pts, src_features, tgt_features, gt_pose]):
                print(f"Skipping batch {batch_idx} due to missing data.")
                continue

            # Move data to the device
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
            corr = corr.to(device)
            labels = labels.to(device)
            xyz_0, xyz_1 = src_pts.to(device), tar_pts.to(device)
            feat_0, feat_1 = src_features.to(device), tgt_features.to(device)
            gt_pose = gt_pose.to(device)

<<<<<<< HEAD
            # xyz_0, xyz_1 = center_and_normalize(xyz_0, xyz_1)
            # print("################@@@@@@@@@@@@@@@@######################")
            # print(xyz_0.shape)
            # print(xyz_1.shape)
            # print(feat_0.shape)
            # print(feat_1.shape)
    
            # Initialize KNN graphs for source and target point clouds
            k = 16
            ####### remove the batch size when it is one ##############
            # Squeeze the first dimension if it's 1 (e.g., torch.Size([1, 2048, 3]) -> torch.Size([2048, 3]))
            if xyz_0.dim() == 3 and xyz_0.size(0) == 1:
                xyz_0 = xyz_0.squeeze(0)
            if xyz_1.dim() == 3 and xyz_1.size(0) == 1:
                xyz_1 = xyz_1.squeeze(0)
            if gt_pose.dim() == 3 and gt_pose.size(0) == 1:
                gt_pose = gt_pose.squeeze(0)
            if corr.dim() == 3 and corr.size(0) == 1:
                corr = corr.squeeze(0)
            if labels.dim() == 3 and labels.size(0) == 1:
                labels = labels.squeeze(0)

            graph_idx_0 = knn_graph(xyz_0, k=k, loop=False)
            graph_idx_1 = knn_graph(xyz_1, k=k, loop=False)


            # Descriptor generation using PointNet or directly using feat_0/feat_1
            if use_pointnet:
                feature_encoder = PointNet().to(device)
                feat_0 = feature_encoder(xyz_0, graph_idx_0, None)  # Descriptor for source
                feat_1 = feature_encoder(xyz_1, graph_idx_1, None)  # Descriptor for target
            # else:
            #     feat_0 = feat_0  # Use pre-computed features directly
            #     feat_1 = feat_1  # Use pre-computed features directly

            if feat_0.dim() == 3 and feat_0.size(0) == 1:
                feat_0 = feat_0.squeeze(0)
            if feat_1.dim() == 3 and feat_1.size(0) == 1:
                feat_1 = feat_1.squeeze(0)

            # feats_0 = feature_encoder(xyz_0, graph_idx_0, None)  # Descriptor for source
            # feats_1 = feature_encoder(xyz_1, graph_idx_1, None)  # Descriptor for target
            # Initialize edges and edge attributes for source and target
            edges_0, edge_attr_0 = get_edges_batch(graph_idx_0, xyz_0.size(0), 1)
            edges_1, edge_attr_1 = get_edges_batch(graph_idx_1, xyz_1.size(0), 1)

            # # Forward pass through the model
            quaternion, translation, corr_loss, h_src_norm, x_src, h_tgt_norm, x_tgt, gt_labels = model(feat_0, xyz_0, edges_0, edge_attr_0, feat_1, xyz_1, edges_1, edge_attr_1, corr, labels)
            #corr_loss
            point_error, feature_loss = compute_losses(quaternion, translation, h_src_norm, x_src, h_tgt_norm, x_tgt, gt_labels)
            # # Compute pose loss
            rot_loss, trans_loss = pose_loss(quaternion, translation, gt_pose, delta=1.5)

            # # Combine pose and correspondence loss
            loss = rot_loss + trans_loss + beta*corr_loss ########### trans_loss + beta*corr_loss

            # Accumulate losses
            running_loss += loss.item()
            running_rot_loss += rot_loss.item()
            running_trans_loss += trans_loss.item()
            running_corr_loss += corr_loss.item()
            running_point_error += point_error.item()
            running_feature_consensus_error += feature_loss.item()

        # Return the average losses over the entire train dataset
        avg_loss = running_loss / len(dataloader)
        avg_rot_loss = running_rot_loss / len(dataloader)
        avg_trans_loss = running_trans_loss / len(dataloader)
        avg_corr_loss = running_corr_loss / len(dataloader)
        avg_point_error = running_point_error / len(dataloader)
        avg_feature_consensus_error = running_feature_consensus_error / len(dataloader)

        print(f'Validation Loss: {avg_loss:.6f} | Pose rot Loss: {avg_rot_loss:.6f} | Pose trans Loss: {avg_trans_loss:.6f} | Point error: {avg_point_error:.6f}, Feature error:  {avg_feature_consensus_error:.6f}')
    # # Log validation losses
    # writer.add_scalar('Loss/validation', avg_loss, epoch)
    # writer.add_scalar('Pose_Loss/validation', avg_pose_loss, epoch)
    # writer.add_scalar('Correspondence_Loss/validation', avg_corr_loss, epoch)
        avg_pose_loss = avg_rot_loss + avg_trans_loss

    return avg_loss, avg_pose_loss, avg_corr_loss, avg_point_error, avg_feature_consensus_error
=======

            # Flatten the input points for KNN computation
            # xyz_0_flat = xyz_0.view(-1, xyz_0.size(-1))  # Shape: [batch_size * num_points, 3]
            # xyz_1_flat = xyz_1.view(-1, xyz_1.size(-1))
            batch_size, num_points, _ = xyz_0.shape

            # Generate batch indices (ensures k-NN only considers points within the same batch)
            batch_indices = torch.arange(batch_size, device=device).repeat_interleave(num_points)  # Shape: [batch_size * num_points]

            # Flatten input points for k-NN computation
            xyz_0_flat = xyz_0.view(-1, 3)  # Shape: [batch_size * num_points, 3]
            xyz_1_flat = xyz_1.view(-1, 3)  # Shape: [batch_size * num_points, 3]

            # Compute k-NN graphs (ensuring edges are within batch instances)
            k = 16
            # graph_idx_0 = knn_graph(xyz_0_i, k=k, loop=False)
            # graph_idx_1 = knn_graph(xyz_1_i, k=k, loop=False)

            graph_idx_0_list = []
            graph_idx_1_list = []

            for i in range(batch_size):
                # Compute kNN graph for each batch
                graph_idx_0_i = knn_graph(xyz_0[i], k=k, loop=True)  # Shape: (2, num_edges)
                graph_idx_1_i = knn_graph(xyz_1[i], k=k, loop=True)  # Shape: (2, num_edges)

                graph_idx_0_list.append(graph_idx_0_i)  # Store edge indices
                graph_idx_1_list.append(graph_idx_1_i)

            # Stack to form (batch_size, 2, num_edges)
            graph_idx_0 = torch.stack(graph_idx_0_list, dim=0)  # Shape: (batch_size, 2, num_edges)
            graph_idx_1 = torch.stack(graph_idx_1_list, dim=0)  # Shape: (batch_size, 2, num_edges)

            # Debugging: Check graph indices
            print(f"graph_idx_0 shape: {graph_idx_0.shape}")  # Expected: (2, batch_size * num_points * k)
            print(f"graph_idx_1 shape: {graph_idx_1.shape}")  # Expected: (2, batch_size * num_points * k)

            # Ensure edges are contained within each batch
            src, dst = graph_idx_0[0], graph_idx_0[1]
            src_batch = batch_indices[src]  # Batch indices for source nodes
            dst_batch = batch_indices[dst]  # Batch indices for target nodes
            cross_batch_edges = (src_batch != dst_batch).sum().item()
            print(f"[DEBUG] Cross-batch edges (should be 0): {cross_batch_edges}")

            # If there are cross-batch edges, raise an error
            if cross_batch_edges > 0:
                raise ValueError("Cross-batch edges detected. Ensure `batch_indices` is correct and `knn_graph` respects batch boundaries.")


            # If using PointNet, encode the features
            if use_pointnet:
                # Use the updated PointNet model
                pointnet = PointNet(in_num_feature=3, hidden_num_feature=32, output_num_feature=32, device=xyz_0.device).to(device)

                # Compute features
                feat_0 = pointnet(xyz_0_flat, graph_idx_0, batch_indices).view(batch_size, num_points, -1)  # [batch_size, 2048, 32]
                feat_1 = pointnet(xyz_1_flat, graph_idx_1, batch_indices).view(batch_size, num_points, -1)  # [batch_size, 2048, 32]

                # feature_encoder = PointNet().to(device)
                # feat_0 = feature_encoder(xyz_0, graph_idx_0, None)
                # feat_1 = feature_encoder(xyz_1, graph_idx_1, None)
            # Reshape to batch-first format
            graph_idx_0 = graph_idx_0.view(batch_size, 2, k*num_points) #.transpose(0, 1)  # Shape: (batch_size, 2, num_edges_per_batch)
            graph_idx_1 = graph_idx_1.view(batch_size, 2, k*num_points) #.transpose(0, 1)  # Shape: (batch_size, 2, num_edges_per_batch)

            edges_0 = None
            edge_attr_0 = None
            edges_1 = None
            edge_attr_1 = None

            # Initialize lists to store the results
            edges_0_list, edge_attr_0_list = [], []
            edges_1_list, edge_attr_1_list = [], []

            # Loop through each batch
            batch_size = xyz_0.size(0)  # Get batch size

            for i in range(batch_size):
                # Extract the i-th batch
                graph_idx_0_i = graph_idx_0[i]  # Shape: [2, 32768]
                graph_idx_1_i = graph_idx_1[i]  # Shape: [2, 32768]
                xyz_0_i = xyz_0[i]  # Shape: [2048, 3]
                xyz_1_i = xyz_1[i]  # Shape: [2048, 3]
                
                # Call get_edges_batch for the i-th batch
                edges_0_i, edge_attr_0_i = get_edges_batch(graph_idx_0_i, xyz_0_i.size(0), 1)
                edges_1_i, edge_attr_1_i = get_edges_batch(graph_idx_1_i, xyz_1_i.size(0), 1)

                # Stack edges correctly: Convert list [row_tensor, col_tensor] → single tensor of shape (2, N)
                edges_0_i = torch.stack(edges_0_i, dim=0)  # Shape: (2, N)
                edges_1_i = torch.stack(edges_1_i, dim=0)  # Shape: (2, N)

                # Store in lists
                edges_0_list.append(edges_0_i)  # Shape: (2, N)
                edge_attr_0_list.append(edge_attr_0_i)  # Shape: (N, 1)
                edges_1_list.append(edges_1_i)  # Shape: (2, N)
                edge_attr_1_list.append(edge_attr_1_i)  # Shape: (N, 1)

            # Stack across the batch dimension
            edges_0 = torch.stack(edges_0_list, dim=0)  # Shape: (batch_size, 2, N)
            edge_attr_0 = torch.stack(edge_attr_0_list, dim=0)  # Shape: (batch_size, N, 1)
            edges_1 = torch.stack(edges_1_list, dim=0)  # Shape: (batch_size, 2, N)
            edge_attr_1 = torch.stack(edge_attr_1_list, dim=0)  # Shape: (batch_size, N, 1)

            print(edges_0.shape)
            print(edge_attr_0.shape)
            print(edges_1.shape)
            print(edge_attr_1.shape)
            # Forward pass through the model
            rot_mat, translation, corr_loss, fea_loss, h_src_norm, x_src, h_tgt_norm, x_tgt, gt_labels = model(feat_0, xyz_0, edges_0, edge_attr_0, feat_1, xyz_1, edges_1, edge_attr_1, corr, labels, gt_pose)

            point_error, feature_loss = compute_losses(rot_mat, translation, h_src_norm, x_src, h_tgt_norm, x_tgt, gt_labels)
            
            # Compute pose loss
            rot_losses, trans_losses = pose_loss(rot_mat, translation, gt_pose, delta=1.5)


            # Combine pose and correspondence loss
            loss = rot_losses.mean() + trans_losses.mean() # + beta * corr_loss.mean()

            # Accumulate losses
            running_loss = running_loss + loss.item()
            running_rot_loss = running_rot_loss + rot_losses.mean().item()
            running_trans_loss = running_trans_loss + trans_losses.mean().item()
            running_corr_loss = running_corr_loss + corr_loss.mean().item()
            running_point_error = running_point_error +  point_error.mean().item()
            running_feat_loss = running_feat_loss + fea_loss.mean().item()
            # running_feature_consensus_error += feature_loss.mean().item()

        # Compute average losses
        num_batches = len(dataloader)
        avg_loss = running_loss / num_batches
        avg_rot_loss = running_rot_loss / num_batches
        avg_trans_loss = running_trans_loss / num_batches

        avg_corr_loss = running_corr_loss / num_batches
        avg_point_error = running_point_error / num_batches
        # avg_feature_consensus_error = running_feature_consensus_error / num_batches

        print(
            f"Validation Loss: {avg_loss:.6f} | Pose rot Loss: {avg_rot_loss:.6f} | Pose trans Loss: {avg_trans_loss:.6f} | Point error: {avg_point_error:.6f}" 
        )     #| Feature error: {avg_feature_consensus_error:.6f}"

        avg_corr_loss = None
    return avg_loss, avg_rot_loss + avg_trans_loss, avg_corr_loss, point_error, feature_loss  #, avg_point_error, avg_feature_consensus_error
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c

# Save the checkpoint
def save_checkpoint(epoch, pointnet, egnn, cross_attention, optimizer, save_dir="./checkpoints", is_best=False, use_pointnet=False):
    """
    Saves the model checkpoint, including PointNet encoder (if used), EGNN layers, CrossAttentionPoseRegression, and optimizer state.

    Args:
        epoch (int): The current epoch number.
        pointnet (torch.nn.Module or None): The PointNet encoder model (optional, depending on use_pointnet).
        egnn (torch.nn.Module): The EGNN layers model.
        cross_attention (torch.nn.Module): The CrossAttentionPoseRegression model.
        optimizer (torch.optim.Optimizer): The optimizer used in training.
        save_dir (str): Directory to save the checkpoints.
        use_pointnet (bool): Whether PointNet encoder is used or not.
    """
    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # Append epoch number to the save file name
    save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
       
    checkpoint = {
        'epoch': epoch,
        'egnn_state_dict': egnn.state_dict(),
        'cross_attention_state_dict': cross_attention.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if use_pointnet and pointnet is not None:
        checkpoint['pointnet_state_dict'] = pointnet.state_dict()

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at epoch {epoch} to {save_path}")

    # If this is the best model so far, save it as the best checkpoint
    if is_best:
        save_path = os.path.join(save_dir, 'best_checkpoint')
        torch.save(checkpoint, save_path)
        print(f"Best Checkpoint saved at epoch {epoch} to {save_path}")


def load_checkpoint(checkpoint_path, pointnet=None, egnn=None, cross_attention=None, optimizer=None, use_pointnet=False, device='cuda:0'):
    """
    Loads a model checkpoint, restoring the model weights, PointNet (if used), EGNN, CrossAttentionPoseRegression, and optimizer state.

    Args:
        checkpoint_path (str): Path to the saved checkpoint.
        pointnet (torch.nn.Module, optional): The PointNet encoder model (optional, depending on use_pointnet).
        egnn (torch.nn.Module, optional): The EGNN layers model (required).
        cross_attention (torch.nn.Module, optional): The CrossAttentionPoseRegression model (required).
        optimizer (torch.optim.Optimizer, optional): The optimizer to restore state (optional).
        use_pointnet (bool): Whether PointNet encoder is used or not.
        device (str): The device to map the model and optimizer (e.g., 'cuda:0' or 'cpu').

    Returns:
        dict: The loaded checkpoint dictionary.
        int: The epoch at which the checkpoint was saved (for resuming training).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found.")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load PointNet encoder weights (if applicable)
    if use_pointnet and pointnet is not None and 'pointnet_state_dict' in checkpoint:
        pointnet.load_state_dict(checkpoint['pointnet_state_dict'])
        print(f"Loaded PointNet encoder weights from {checkpoint_path}")

    # Load EGNN weights
    if egnn is not None and 'egnn_state_dict' in checkpoint:
        egnn.load_state_dict(checkpoint['egnn_state_dict'])
        print(f"Loaded EGNN weights from {checkpoint_path}")
    
    # Load CrossAttentionPoseRegression weights
    if cross_attention is not None and 'cross_attention_state_dict' in checkpoint:
        cross_attention.load_state_dict(checkpoint['cross_attention_state_dict'])
        print(f"Loaded CrossAttentionPoseRegression weights from {checkpoint_path}")

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded optimizer state from {checkpoint_path}")

    # Return the epoch to resume training from
    return checkpoint, checkpoint.get('epoch', 0)

<<<<<<< HEAD
# Main training loop
=======
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, writer, use_pointnet=False, log_interval=10, beta=0.1, save_path="./checkpoints"):
    """
    Train the model and save checkpoints during training.

    Args:
        model (torch.nn.Module): CrossAttentionPoseRegression model which contains EGNN.
        train_loader (torch.utils.data.DataLoader): Dataloader for training data.
        val_loader (torch.utils.data.DataLoader): Dataloader for validation data.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        device (torch.device): Device to train on (e.g., 'cuda' or 'cpu').
        use_pointnet (bool): Whether to use PointNet encoder.
        log_interval (int): Interval for logging the training progress.
        save_path (str): Path to save the model checkpoints.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
<<<<<<< HEAD
    # Create Adam optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler for decaying the learning rate every 15 epochs
    scheduler = StepLR(optimizer, step_size=25, gamma=0.1)  # Gamma defines the decay factor

    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    best_val_loss = float('inf')  # Track the best validation loss
    val_loss = float('inf')
    # If using PointNet encoder, initialize it separately
    if use_pointnet:
        pointnet = PointNet().to(device)
    else:
        pointnet = None  # No PointNet in this case

    # Extract EGNN and CrossAttentionPoseRegression components from the model
    egnn = model.egnn
    cross_attention = model  # CrossAttentionPoseRegression is the top-level model

    for epoch in range(num_epochs):
        # Train for one epoch
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer, use_pointnet, log_interval, beta)
        
        # Step the scheduler
        scheduler.step()

        # Print current learning rate for monitoring
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{num_epochs}, Current LR: {current_lr}")

        # Validate every few epochs (e.g., every 5 epochs)
        if (epoch + 1) % 1 == 0:
            val_loss, val_pose_loss, val_corr_loss, avg_point_error, avg_feature_consensus_error = validate(model, val_loader, device, epoch, writer, use_pointnet)
            print(val_loss)
            print(f'Epoch {epoch + 1}/{num_epochs} - Eva Avg Loss: {val_loss:.6f}, Validation Avg Loss: {val_pose_loss:.6f}, Val corr loss: {val_corr_loss: 6f}')
        else:
            print(f'Epoch {epoch + 1}/{num_epochs} - Eva Avg Loss: {val_loss:.6f}')

        # Save the checkpoint if the validation loss is the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch + 1, pointnet, egnn, cross_attention, optimizer, save_path, is_best=True, use_pointnet=use_pointnet)
            print(f'Best model checkpoint saved at epoch {epoch + 1} with validation loss: {best_val_loss:.6f}')

        # Save checkpoint every 16 epochs
        if (epoch + 1) % 20 == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = save_path  #"./checkpoints"
            save_checkpoint(epoch + 1, pointnet, egnn, cross_attention, optimizer, checkpoint_path, use_pointnet=use_pointnet)

def evaluate_model(checkpoint_path, model, dataloader, device, use_pointnet=False):
    """
    Evaluate the model using a saved checkpoint, and print the predicted pose as a homogeneous transform matrix.

    Args:
        checkpoint_path (str): Path to the checkpoint to be loaded.
        model (torch.nn.Module): The model to be evaluated.
        dataloader (torch.utils.data.DataLoader): Dataloader for the evaluation data.
        device (torch.device): Device to run evaluation on (e.g., 'cuda' or 'cpu').
        use_pointnet (bool): Whether to use PointNet encoder.

    Returns:
        float: The average loss over the evaluation dataset.
    """
    # Load the checkpoint and restore the model's weights
    egnn = model.egnn
    cross_attention = model  # CrossAttentionPoseRegression is the top-level model

    # If using PointNet, define the PointNet encoder
=======
    scheduler = StepLR(optimizer, step_size=200, gamma=0.5)

    best_val_loss = float('inf')

    if use_pointnet:
        pointnet = PointNet().to(device)
    else:
        pointnet = None

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer, use_pointnet, log_interval, beta)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{num_epochs}, Current LR: {current_lr}")

        if (epoch + 1) % 1 == 0:
            val_loss, val_pose_loss, val_corr_loss, avg_point_error, avg_feature_consensus_error = validate(model, val_loader, device, epoch, writer, use_pointnet)
            print(f'Epoch {epoch + 1}/{num_epochs} - Validation Loss: {val_loss:.6f}')
        else:
            print(f'Epoch {epoch + 1}/{num_epochs} - Validation Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch + 1, pointnet, model.egnn, model, optimizer, save_path, is_best=True, use_pointnet=use_pointnet)
            print(f'Best model checkpoint saved at epoch {epoch + 1} with validation loss: {best_val_loss:.6f}')

        if (epoch + 1) % 20 == 0 or (epoch + 1) == num_epochs:
            save_checkpoint(epoch + 1, pointnet, model.egnn, model, optimizer, save_path, use_pointnet=use_pointnet)

def evaluate_model(checkpoint_path, model, dataloader, device, use_pointnet=False):
    egnn = model.egnn
    cross_attention = model

>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
    if use_pointnet:
        pointnet = PointNet().to(device)
    else:
        pointnet = None

<<<<<<< HEAD
    # Load the checkpoint
    _, epoch = load_checkpoint(checkpoint_path, pointnet, egnn, cross_attention, optimizer=None, use_pointnet=use_pointnet, device=device)
    print(f"Checkpoint loaded from epoch {epoch}")

    # Set the model to evaluation mode
    model.eval()

    running_loss = 0.0
    running_corr_loss = 0.0  # Track correspondence rank loss
    running_rot_loss = 0.0  # Track pose rot loss
    running_trans_loss = 0.0  # Track pose trans loss
    running_point_error = 0.0
    running_feature_consensus_error = 0.0

    with torch.no_grad():  # No need to track gradients during validation
        for batch_idx, (corr, labels, src_pts, tar_pts, src_features, tgt_features, gt_pose) in enumerate(dataloader):
            # Check if any returned data is None, and skip if so
            if corr is None or labels is None or src_pts is None or tar_pts is None or src_features is None or tgt_features is None or gt_pose is None:
                print(f"Skipping batch {batch_idx} due to missing data.")
                continue  # Skip this batch

            # Move the data to the specified device (GPU/CPU)
=======
    _, epoch = load_checkpoint(checkpoint_path, pointnet, egnn, cross_attention, optimizer=None, use_pointnet=use_pointnet, device=device)
    print(f"Checkpoint loaded from epoch {epoch}")

    model.eval()
    total_loss = 0.0
    total_pose_loss = 0.0
    total_corr_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (corr, labels, src_pts, tar_pts, src_features, tgt_features, gt_pose) in enumerate(dataloader):
            if corr is None or labels is None or src_pts is None or tar_pts is None or src_features is None or tgt_features is None or gt_pose is None:
                continue

>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
            corr = corr.to(device)
            labels = labels.to(device)
            xyz_0, xyz_1 = src_pts.to(device), tar_pts.to(device)
            feat_0, feat_1 = src_features.to(device), tgt_features.to(device)
            gt_pose = gt_pose.to(device)
<<<<<<< HEAD
    
            # Initialize KNN graphs for source and target point clouds
            k = 16
            ####### remove the batch size when it is one ##############
            # Squeeze the first dimension if it's 1 (e.g., torch.Size([1, 2048, 3]) -> torch.Size([2048, 3]))
            if xyz_0.dim() == 3 and xyz_0.size(0) == 1:
                xyz_0 = xyz_0.squeeze(0)
            if xyz_1.dim() == 3 and xyz_1.size(0) == 1:
                xyz_1 = xyz_1.squeeze(0)
            if gt_pose.dim() == 3 and gt_pose.size(0) == 1:
                gt_pose = gt_pose.squeeze(0)
            if corr.dim() == 3 and corr.size(0) == 1:
                corr = corr.squeeze(0)
            if labels.dim() == 3 and labels.size(0) == 1:
                labels = labels.squeeze(0)

            graph_idx_0 = knn_graph(xyz_0, k=k, loop=False)
            graph_idx_1 = knn_graph(xyz_1, k=k, loop=False)


            # Descriptor generation using PointNet or directly using feat_0/feat_1
            if use_pointnet:
                feature_encoder = PointNet().to(device)
                feat_0 = feature_encoder(xyz_0, graph_idx_0, None)  # Descriptor for source
                feat_1 = feature_encoder(xyz_1, graph_idx_1, None)  # Descriptor for target
            # else:
            #     feat_0 = feat_0  # Use pre-computed features directly
            #     feat_1 = feat_1  # Use pre-computed features directly

            if feat_0.dim() == 3 and feat_0.size(0) == 1:
                feat_0 = feat_0.squeeze(0)
            if feat_1.dim() == 3 and feat_1.size(0) == 1:
                feat_1 = feat_1.squeeze(0)

            # feats_0 = feature_encoder(xyz_0, graph_idx_0, None)  # Descriptor for source
            # feats_1 = feature_encoder(xyz_1, graph_idx_1, None)  # Descriptor for target
            # Initialize edges and edge attributes for source and target
            edges_0, edge_attr_0 = get_edges_batch(graph_idx_0, xyz_0.size(0), 1)
            edges_1, edge_attr_1 = get_edges_batch(graph_idx_1, xyz_1.size(0), 1)

            # # Forward pass through the model
            quaternion, translation, corr_loss, h_src_norm, x_src, h_tgt_norm, x_tgt, gt_labels = model(feat_0, xyz_0, edges_0, edge_attr_0, feat_1, xyz_1, edges_1, edge_attr_1, corr, labels)
            #corr_loss
            point_error, feature_loss = compute_losses(quaternion, translation, h_src_norm, x_src, h_tgt_norm, x_tgt, gt_labels)
            # # Compute pose loss
            rot_loss, trans_loss = pose_loss(quaternion, translation, gt_pose, delta=1.5)

            # # Combine pose and correspondence loss
            loss = rot_loss + trans_loss + beta*corr_loss ########### trans_loss + beta*corr_loss

            # Accumulate losses
            running_loss += loss.item()
            running_rot_loss += rot_loss.item()
            running_trans_loss += trans_loss.item()
            running_corr_loss += corr_loss.item()
            running_point_error += point_error.item()
            running_feature_consensus_error += feature_loss.item()

    # Return the average losses over the entire train dataset
    avg_loss = running_loss / len(dataloader)
    avg_rot_loss = running_rot_loss / len(dataloader)
    avg_trans_loss = running_trans_loss / len(dataloader)
    avg_corr_loss = running_corr_loss / len(dataloader)
    avg_point_error = running_point_error / len(dataloader)
    avg_feature_consensus_error = running_feature_consensus_error / len(dataloader)

    print(f"Evaluation completed. Avg Loss: {avg_loss:.6f}, Avg Rot Loss: {avg_rot_loss:.6f}, Avg Trans Loss: {avg_trans_loss:.6f} Avg Corr Loss: {avg_corr_loss:.6f}")
    
    return avg_loss, avg_rot_loss, avg_trans_loss, avg_corr_loss
=======

            k = 12
            graph_idx_0 = knn_graph(xyz_0, k=k, loop=False)
            graph_idx_1 = knn_graph(xyz_1, k=k, loop=False)

            if use_pointnet:
                feat_0 = pointnet(xyz_0, graph_idx_0, None)
                feat_1 = pointnet(xyz_1, graph_idx_1, None)

            # Initialize lists to store the results
            edges_0_list, edge_attr_0_list = [], []
            edges_1_list, edge_attr_1_list = [], []

            # Loop through each batch
            batch_size = graph_idx_0.size(0)  # Get batch size
            for i in range(batch_size):
                # Extract the i-th batch
                graph_idx_0_i = graph_idx_0[i]  # Shape: [2, 32768]
                graph_idx_1_i = graph_idx_1[i]  # Shape: [2, 32768]
                xyz_0_i = xyz_0[i]  # Shape: [2048, 3]
                xyz_1_i = xyz_1[i]  # Shape: [2048, 3]

                # Call get_edges_batch for the i-th batch
                edges_0_i, edge_attr_0_i = get_edges_batch(graph_idx_0_i, xyz_0_i.size(0), 1)  # Shapes: [2, num_edges], [num_edges, edge_attr_dim]
                edges_1_i, edge_attr_1_i = get_edges_batch(graph_idx_1_i, xyz_1_i.size(0), 1)  # Shapes: [2, num_edges], [num_edges, edge_attr_dim]

                # Append the results to the lists
                edges_0_list.append(edges_0_i)
                edge_attr_0_list.append(edge_attr_0_i)
                edges_1_list.append(edges_1_i)
                edge_attr_1_list.append(edge_attr_1_i)

            # Stack the results back into batch tensors
            edges_0 = torch.stack(edges_0_list, dim=0)  # Shape: [batch_size, 2, num_edges]
            edge_attr_0 = torch.stack(edge_attr_0_list, dim=0)  # Shape: [batch_size, num_edges, edge_attr_dim]
            edges_1 = torch.stack(edges_1_list, dim=0)  # Shape: [batch_size, 2, num_edges]
            edge_attr_1 = torch.stack(edge_attr_1_list, dim=0)  # Shape: [batch_size, num_edges, edge_attr_dim]

            rot_mat, translation, corr_loss, fea_loss, h_src_norm, x_src, h_tgt_norm, x_tgt, gt_labels = model(feat_0, xyz_0, edges_0, edge_attr_0, feat_1, xyz_1, edges_1, edge_attr_1, corr, labels, gt_pose)

            for i in range(rot_mat.size(0)):
                rot_matrix = rot_mat[i]     #####quaternion_to_matrix(quaternion[i], device=device)
                trans_vector = translation[i]

                hom_matrix = torch.eye(4, device=device)
                hom_matrix[:3, :3] = rot_matrix
                hom_matrix[:3, 3] = trans_vector

                print(f"Predicted Pose (Batch {batch_idx}, Item {i}):")
                print(hom_matrix)

            pose_losses = pose_loss(rot_mat, translation, gt_pose, delta=1.5)
            loss = pose_losses + corr_loss
            total_loss = total_loss + loss.item()
            total_pose_loss = total_pose_loss + pose_losses.item()
            total_corr_loss = total_corr_loss + corr_loss.item()
            num_batches = num_batches +  1

    avg_loss = total_loss / num_batches
    avg_pose_loss = total_pose_loss / num_batches
    avg_corr_loss = total_corr_loss / num_batches

    print(f"Evaluation completed. Avg Loss: {avg_loss:.6f}, Avg Pose Loss: {avg_pose_loss:.6f}, Avg Corr Loss: {avg_corr_loss:.6f}")
    return avg_loss, avg_pose_loss, avg_corr_loss
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c

def get_args():
    parser = argparse.ArgumentParser(description="Training a Pose Regression Model")
    
    # Add arguments with default values
<<<<<<< HEAD
    parser.add_argument('--base_dir', type=str, default='/home/eavise3d/3DMatch_FCGF_Feature_32_transform', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--num_node', type=int, default=2048, help='Number of nodes in the graph')
    parser.add_argument('--k', type=int, default=12, help='Number of nearest neighbors in KNN graph')
    parser.add_argument('--in_node_nf', type=int, default=32, help='Input feature size for EGNN')
    parser.add_argument('--hidden_node_nf', type=int, default=64, help='Hidden node feature size for EGNN')
    parser.add_argument('--sim_hidden_nf', type=int, default=35, help='Hidden dimension after concatenation in EGNN')
    parser.add_argument('--out_node_nf', type=int, default=32, help='Output node feature size for EGNN')
=======
    parser.add_argument('--base_dir', type=str, default="/media/eavise3d/新加卷/Datasets/eccv-data-0126/kitti/kitti/dataset", help='Path to the dataset')  ### "/media/eavise3d/新加卷/Datasets/luan-scripts/3DMatch/3DMatch"
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--num_node', type=int, default=2048, help='Number of nodes in the graph')
    parser.add_argument('--k', type=int, default=12, help='Number of nearest neighbors in KNN graph')
    parser.add_argument('--in_node_nf', type=int, default=33, help='Input feature size for EGNN')
    parser.add_argument('--hidden_node_nf', type=int, default=33, help='Hidden node feature size for EGNN') ### fpfh 33 fcgf 32
    parser.add_argument('--sim_hidden_nf', type=int, default=33, help='Hidden dimension after concatenation in EGNN')
    parser.add_argument('--out_node_nf', type=int, default=33, help='Output node feature size for EGNN')
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in EGNN')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "val"], help='Mode to run the model (train/val)')
    parser.add_argument('--lossBeta', type=float, default=1e-2, help='Correspondence loss weights')
    parser.add_argument('--savepath', type=str, default='./checkpoints/model_checkpoint.pth', help='Path to the dataset')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
<<<<<<< HEAD
    # print("###edge batch begins 2###########")
=======

    # print("###edge batch begins ###########")
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
    # # Set up the data and training parameters
    # Load the arguments
    base_dir = args.base_dir
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    num_node = args.num_node
    k = args.k

    in_node_nf = args.in_node_nf
    hidden_node_nf = args.hidden_node_nf
    sim_hidden_nf = args.sim_hidden_nf
    out_node_nf = args.out_node_nf
    n_layers = args.n_layers
    beta = args.lossBeta
    mode = args.mode
    savepath = args.savepath

    mode = "train" ### set to "eval" for inference mode
<<<<<<< HEAD

    train_dataset = ThreeDMatchTrainVal(root=base_dir, 
                            split=mode,   
                            descriptor='fcgf',
=======
    train_dataset = KITTItrainVal(root=base_dir,   ####  ThreeDMatchTrainVal
                            split=mode,   
                            descriptor='fpfh',
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
                            in_dim=6,
                            inlier_threshold=0.10,
                            num_node=2048, 
                            use_mutual=True,
                            downsample=0.03, 
                            augment_axis=1, 
                            augment_rotation=1.0,
                            augment_translation=0.01,
                        )

<<<<<<< HEAD
    val_dataset = ThreeDMatchTrainVal(root=base_dir, 
                            split='val',   
                            descriptor='fcgf',
=======
    val_dataset = KITTItrainVal(root=base_dir,    ### ThreeDMatchTrainVal
                            split='val',   
                            descriptor='fpfh',
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
                            in_dim=6,
                            inlier_threshold=0.10,
                            num_node=2048, 
                            use_mutual=True,
                            downsample=0.03, 
                            augment_axis=1, 
                            augment_rotation=1.0,
                            augment_translation=0.01,
                        )

<<<<<<< HEAD
    test_dataset = ThreeDMatchTest(root=base_dir, 
                            split='test',   
                            descriptor='fcgf',
=======
    test_dataset =  KITTItest(root=base_dir,      ###  ThreeDMatchTest
                            split='test',   
                            descriptor='fpfh',
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
                            in_dim=6,
                            inlier_threshold=0.10,
                            num_node=2048, 
                            use_mutual=True,
                            downsample=0.03, 
                            augment_axis=1, 
                            augment_rotation=1.0,
                            augment_translation=0.01,
                        )
    # Instantiate the dataset

<<<<<<< HEAD

=======
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
    # Initialize TensorBoard writer
    # Ensure the directory exists
    if not os.path.exists('runs/pose_regression_experiment'):
        os.makedirs('runs/pose_regression_experiment')
        print("Created directory for log init runs/pose_regression_experiment")
    writer = SummaryWriter(log_dir='runs/pose_regression_experiment')

    # # Create DataLoaders
    if mode == "train":
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
<<<<<<< HEAD
    elif mode == "test":
=======
    elif mode == "eval":
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    egnn = EGNN(in_node_nf=in_node_nf, hidden_nf=hidden_node_nf, out_node_nf=out_node_nf, in_edge_nf=1, n_layers=n_layers)
    egnn.to(dev)
    # egnn(h, x, edges1, edge_attr1)
    # print("###edge batch ends###########")
    cross_attention_model = CrossAttentionPoseRegression(egnn=egnn, num_nodes=num_node, hidden_nf=sim_hidden_nf, device=dev).to(dev)
    cross_attention_model.to(dev)

    # cross_attention_model(h, x, graph_idx1, edge_attr1, h, x, graph_idx1, edge_attr1)
    ##########comment these lines during evaluation mode#################
    if mode == "train":
        train_model(cross_attention_model, train_loader, val_loader, num_epochs=num_epochs, \
                learning_rate=learning_rate, device=dev, writer=writer, use_pointnet=False, log_interval=10, beta=0.1, save_path=savepath)
    elif mode == "test":
        checkpoint_path = "./checkpoints/model_epoch_16.pth" #####specify the right path of the saved checkpint#######
<<<<<<< HEAD
        avg_loss, avg_rot_loss, avg_trans_loss, avg_corr_loss = evaluate_model(checkpoint_path, cross_attention_model, val_loader, device=dev, use_pointnet=False)
=======
        avg_loss, avg_pose_loss, avg_corr_loss = evaluate_model(checkpoint_path, cross_attention_model, val_loader, device=dev, use_pointnet=True)
>>>>>>> 2a357122e30aab6fa1881427d94545722bf9864c
