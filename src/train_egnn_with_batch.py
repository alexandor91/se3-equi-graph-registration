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


class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='max')  # "max" aggregation
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Start propagating messages
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_j: torch.Tensor, pos_j: torch.Tensor, pos_i: torch.Tensor) -> torch.Tensor:
        # Concatenate features with relative positional differences
        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)


class PointNet(nn.Module):
    def __init__(self, in_num_feature=3, hidden_num_feature=32, output_num_feature=32, device='cuda:0'):
        super().__init__()
        self.hidden_num_feature = hidden_num_feature
        self.conv1 = PointNetLayer(in_num_feature, hidden_num_feature)
        self.conv2 = PointNetLayer(hidden_num_feature, output_num_feature)
        self.device = device

    def forward(self, pos: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Perform two layers of message passing
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
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
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, num_heads=4,
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
    def __init__(self, egnn, num_nodes=2048, hidden_nf=32, device='cuda:0'):
        super(CrossAttentionPoseRegression, self).__init__()
        self.egnn = egnn
        self.hidden_nf = hidden_nf
        self.num_nodes = num_nodes
        self.device = device

        # MLP layers for compression
        self.mlp_compress = nn.Sequential(
            nn.Linear(num_nodes, num_nodes // 2),
            nn.ReLU(),
            nn.Linear(num_nodes // 2, num_nodes // 4),
            nn.ReLU(),
            nn.Linear(num_nodes // 4, 128)
        )

        # MLP layers for pose regression
        self.mlp_pose = nn.Sequential(
            nn.Linear(128 * 2 * 3, 256), # self.hidden_nf
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
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
        self.bn1 = nn.BatchNorm1d(self.hidden_nf)  # Normalize feature dimension
        self.bn2 = nn.BatchNorm1d((self.hidden_nf+3))  # Normalize feature dimension
        # Weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # self.mlp_compress.apply(init_layer)
        self.mlp_pose.apply(init_layer)
        # self.shared_mlp_decoder.apply(init_layer)
        # self.shallow_mlp_pose.apply(init_layer)

    def forward(self, h_src, x_src, edges_src, edge_attr_src, h_tgt, x_tgt, edges_tgt, edge_attr_tgt, corr, labels):
        batch_size, num_points, feature_dim = h_src.shape
        device = h_src.device

        # Row-wise dot product between h_src and h_tgt (resulting in B x N x 1)
        similarity_scores = torch.sum(h_src * h_tgt, dim=-1, keepdim=True)  # [B, N, 1]

        # Get top-k scores and indices (k=128)
        top_scores, top_indices = torch.topk(similarity_scores.squeeze(-1), k=128, dim=-1)  # [B, 128]

        # Gather top-k features and corresponding point coordinates
        batch_indices = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, 128)
        compressed_h_src = torch.gather(h_src, dim=1, index=top_indices.unsqueeze(-1).expand(-1, -1, feature_dim))  # [B, 128, 32]
        compressed_h_tgt = torch.gather(h_tgt, dim=1, index=top_indices.unsqueeze(-1).expand(-1, -1, feature_dim))  # [B, 128, 32]

        compressed_x_src = torch.gather(x_src, dim=1, index=top_indices.unsqueeze(-1).expand(-1, -1, 3))  # [B, 128, 3]
        compressed_x_tgt = torch.gather(x_tgt, dim=1, index=top_indices.unsqueeze(-1).expand(-1, -1, 3))  # [B, 128, 3]

        B, N, _ = x_src.shape

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

            weight_scores = torch.sum(valid_src_features * valid_tgt_features, dim=-1)  # Shape: (batch_size, N)

            # Optional: Normalize weight scores (e.g., softmax across rows for each batch)
            weights = torch.nn.functional.softmax(weight_scores, dim=-1)  # Shape: (batch_size, N)

            # Compute weighted centroids
            src_centroid = (weights[:, None] * valid_src_points).sum(dim=0, keepdim=True)  # (1, 3)
            tgt_centroid = (weights[:, None] * valid_tgt_points).sum(dim=0, keepdim=True)  # (1, 3)

            # Centralize points
            src_centered = valid_src_points - src_centroid  # (N_valid, 3)
            tgt_centered = valid_tgt_points - tgt_centroid  # (N_valid, 3)

            # weighted cross-covariance matrix
            H = (weights[:, None, None] * src_centered[:, :, None] @ tgt_centered[:, None, :]).sum(dim=0)  # (3, 3)

            # erform SVD
            U, S, Vt = torch.linalg.svd(H)
            R_b = Vt.T @ U.T

            # Ensure proper rotation (det(R) = 1)
            if torch.det(R_b) < 0:
                Vt[-1, :] *= -1
                R_b = Vt.T @ U.T

            # Compute translation
            t_b = tgt_centroid.squeeze() - R_b @ src_centroid.squeeze()
            # Store results
            R[b] = R_b
            t[b] = t_b

        # # Concatenate selected source and target coordinates
        # combined_coordinates = torch.cat([compressed_x_src, compressed_x_tgt], dim=-1)  # [B, 128, 6]
        # flattened_features = combined_coordinates.view(batch_size, -1)  # [B, 128 * 64]

        # # Predict pose using MLP decoder
        # pose = self.mlp_pose(flattened_features)  # [B, 128, 7] (4 for quaternion, 3 for translation)

        # quaternion = F.normalize(pose[:, :4], p=2, dim=-1)
        # # quaternion = rotation_matrix_to_quaternion_batch(R)

        # translation = pose[:, 4:]

        # Step 8: Combine SVD output into a 4x4 pose matrix
        initial_pose = torch.eye(4, device=x_src.device).repeat(B, 1, 1)  # [B, 4, 4]
        initial_pose[:, :3, :3] = R
        initial_pose[:, :3, 3] = t

        # Step 9: Prepare input for MLP refinement
        combined_coordinates = torch.cat([compressed_x_src, compressed_x_tgt], dim=-1)  # [B, N, 6]
        flattened_features = combined_coordinates.view(B, -1)  # [B, N * 6]

        # Predict residual pose using MLP decoder
        delta_pose = self.mlp_pose(flattened_features)  # [B, 7]
        delta_quaternion = F.normalize(delta_pose[:, :4], p=2, dim=-1)  # Normalize quaternion
        delta_translation = delta_pose[:, 4:]  # [B, 3]

        # Convert delta_quaternion to rotation matrix
        delta_rotation = quaternion_to_matrix(delta_quaternion)

        # Apply residual pose refinement
        refined_R = torch.bmm(delta_rotation, R)  # Refine rotation
        refined_t = t + delta_translation  # Refine translation

        # Assemble refined pose
        refined_pose = torch.eye(4, device=x_src.device).repeat(B, 1, 1)  # [B, 4, 4]
        refined_pose[:, :3, :3] = refined_R
        refined_pose[:, :3, 3] = refined_t

        quaternion = F.normalize(rotation_matrix_to_quaternion_batch(refined_pose[:, :4]), p=2, dim=-1)
        # quaternion = rotation_matrix_to_quaternion_batch(R)

        translation = refined_pose[:, :3, 3]
        # translation = t
        # Optionally, compute loss
        # svd_loss = self.compute_svd_loss(compressed_h_src, compressed_h_tgt)
        # Combine all losses (if needed)
        total_loss = None

        return quaternion, translation, total_loss, h_src, x_src, h_tgt, x_tgt, labels

    
def compute_losses(quaternion, translation, h_src_norm, x_src, h_tgt_norm, x_tgt, gt_labels):
    """
    Args:
        quaternion: Tensor of shape [batch_size, 4] (predicted quaternion).
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
    quaternion = F.normalize(quaternion, p=2, dim=-1)  # Normalize quaternion

    batch_size = quaternion.shape[0]

    # Convert quaternion to rotation matrices
    rotation_matrices = quaternion_to_matrix(quaternion)
    # torch.stack([
    #     quaternion_to_matrix(quaternion[b]).to(device)
    #     for b in range(batch_size)
    # ], dim=0)  # Shape: [batch_size, 3, 3]

    # Transform source points
    x_src_transformed = torch.matmul(rotation_matrices, x_src.transpose(1, 2)).transpose(1, 2) + translation.unsqueeze(1)

    # Compute point distances
    point_distances = torch.norm(x_src_transformed - x_tgt, dim=-1)  # Shape: [batch_size, N]

    # Apply mask for valid correspondences
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

    # Compute feature loss
    feature_loss = torch.mean(
        torch.norm(
            h_src_norm[gt_labels == 1] - h_tgt_norm[gt_labels == 1],
            dim=-1
        )
    )

    return point_error, feature_loss

def pose_loss(pred_quaternion, pred_translation, gt_pose, delta=1.5):
    """
    Compute the loss between the predicted pose (quaternion + translation) and the ground truth pose matrix.
    
    Arguments:
    - pred_quaternion: Predicted quaternion of shape [B, 4]
    - pred_translation: Predicted translation of shape [B, 3]
    - gt_pose: Ground truth pose as a batch of 4x4 matrices, shape [B, 4, 4]
    - delta: Huber loss delta parameter
    
    Returns:
    - Rotation loss: Tensor of shape [B]
    - Translation loss: Tensor of shape [B]
    """
    batch_size = pred_quaternion.size(0)
    
    # Extract ground truth translation and rotation
    gt_translation = gt_pose[:, :3, 3]  # Shape: [B, 3]
    gt_rotation = gt_pose[:, :3, :3]    # Shape: [B, 3, 3]
    gt_quaternion = rotation_matrix_to_quaternion_batch(gt_rotation)  # Convert [B, 3, 3] to [B, 4]

    # Normalize ground truth and predicted quaternions
    gt_quaternion = F.normalize(gt_quaternion, p=2, dim=-1)      # Shape: [B, 4]
    pred_quaternion = F.normalize(pred_quaternion, p=2, dim=-1)  # Shape: [B, 4]

    # Convert predicted quaternion to rotation matrix
    pred_rotation = quaternion_to_matrix(pred_quaternion)  # Shape: [B, 3, 3]

    # print("@@@@@@@@@@@@@@")
    # print(pred_rotation.shape)
    # print(gt_rotation.shape)
    # Compute geodesic rotation loss
    # Compute the batched rotation loss
    # Compute rotation loss
    pred_rotation_3x3 = pred_rotation[:, :3, :3]
    gt_rotation_3x3 = gt_rotation[:, :3, :3]

    pred_rotation_T = pred_rotation_3x3.transpose(-1, -2)
    R = torch.matmul(pred_rotation_T, gt_rotation_3x3)
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

        batch_size, num_points, _ = xyz_0.size()

        # Flatten the input points for KNN computation
        # xyz_0_flat = xyz_0.view(-1, xyz_0.size(-1))  # Shape: [batch_size * num_points, 3]
        # xyz_1_flat = xyz_1.view(-1, xyz_1.size(-1))

        # # Create the batch tensor for KNN graph
        # batch_tensor = torch.arange(batch_size, device=device).repeat_interleave(num_points)

        # Compute KNN graphs
        # Input: xyz_0 and xyz_1 are tensors of shape [batch_size, 2048, 3]
        batch_size, num_points, _ = xyz_0.shape

        # Prepare batch indices for k-NN computation
        batch_indices = torch.arange(batch_size).repeat_interleave(num_points).to(xyz_0.device)  # [batch_size * 2048]

        # Flatten the batch dimensions
        xyz_0_flat = xyz_0.view(-1, 3)  # Shape: [batch_size * 2048, 3]
        xyz_1_flat = xyz_1.view(-1, 3)  # Shape: [batch_size * 2048, 3]

        # Construct k-NN graphs while keeping points from different batches independent
        k = 32
        graph_idx_0 = knn_graph(xyz_0_flat, k=k, batch=batch_indices, loop=False)
        graph_idx_1 = knn_graph(xyz_1_flat, k=k, batch=batch_indices, loop=False)

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

        # # Generate edges and edge attributes
        # edges_0, edge_attr_0 = get_edges_batch(graph_idx_0, xyz_0_flat.size(0), 1)
        # edges_1, edge_attr_1 = get_edges_batch(graph_idx_1, xyz_1_flat.size(0), 1)
        edges_0 = None
        edge_attr_0 = None
        edges_1 = None
        edge_attr_1 = None

        # Forward pass through the model
        quaternion, translation, corr_loss, h_src_norm, x_src, h_tgt_norm, x_tgt, gt_labels = model(feat_0, xyz_0, edges_0, edge_attr_0, feat_1, xyz_1, edges_1, edge_attr_1, corr, labels)

        point_error, feature_loss = compute_losses(quaternion, translation, h_src_norm, x_src, h_tgt_norm, x_tgt, gt_labels)
        
        # Compute pose loss
        rot_losses, trans_losses = pose_loss(quaternion, translation, gt_pose, delta=1.5)

        # Compute the mean loss for the batch
        # mean_loss = rot_losses.mean()
        # std_loss = rot_losses.std(unbiased=False) + 1e-6  # Add small value to avoid division by zero
        # normalized_batch_losses = (rot_losses - mean_loss) / std_loss

        # # Optionally scale normalized losses
        # scaled_losses = normalized_batch_losses * 1.0  # Scaling factor if desired

        # # Compute final loss for backpropagation (mean of normalized losses)
        # rot_loss_mean = torch.mean(scaled_losses)

        # mean_trans_loss = trans_losses.mean()
        # std_trans_loss = trans_losses.std(unbiased=False) + 1e-6  # Add small value to avoid division by zero
        # normalized_batch_trans_losses = (trans_losses - mean_trans_loss) / std_trans_loss

        # # Optionally scale normalized losses
        # scaled_trans_losses = normalized_batch_trans_losses * 1.0  # Scaling factor if desired

        # # Compute final loss for backpropagation (mean of normalized losses)
        # trans_loss_mean = torch.mean(scaled_trans_losses)
        rot_loss_mean = rot_losses.mean()  # Normalize rotation loss across the batch
        trans_loss_mean = trans_losses.mean()  # Normalize translation loss across the batch

        # Combine the normalized losses if needed
        total_loss = rot_loss_mean + trans_loss_mean + point_error 
        # print("#####################")
        # print(rot_loss_mean)
        # print(trans_loss_mean)

        # Combine pose and correspondence losses
        loss = total_loss #+ beta * corr_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            print(f'Epoch [{epoch + 1}], Batch [{batch_idx + 1}], Loss: {loss.item():.6f}')

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}], Average Loss: {avg_loss:.6f}')
    return avg_loss


# Function to validate the model on the validation set
def validate(model, dataloader, device, epoch, writer, use_pointnet=False, beta=1):
    model.eval()
    running_loss = 0.0
    running_corr_loss = 0.0
    running_rot_loss = 0.0
    running_trans_loss = 0.0
    running_point_error = 0.0
    running_feature_consensus_error = 0.0

    with torch.no_grad():
        for batch_idx, (corr, labels, src_pts, tar_pts, src_features, tgt_features, gt_pose) in enumerate(dataloader):
            if any(x is None for x in [corr, labels, src_pts, tar_pts, src_features, tgt_features, gt_pose]):
                print(f"Skipping batch {batch_idx} due to missing data.")
                continue

            # Move data to the device
            corr = corr.to(device)
            labels = labels.to(device)
            xyz_0, xyz_1 = src_pts.to(device), tar_pts.to(device)
            feat_0, feat_1 = src_features.to(device), tgt_features.to(device)
            gt_pose = gt_pose.to(device)

            # # Compute KNN graphs
            # k = 16
            # graph_idx_0 = knn_graph(
            #     xyz_0.view(-1, xyz_0.size(-1)), k=k, loop=False, batch=torch.repeat_interleave(xyz_0.shape[1], xyz_0.shape[0])
            # )
            # graph_idx_1 = knn_graph(
            #     xyz_1.view(-1, xyz_1.size(-1)), k=k, loop=False, batch=torch.repeat_interleave(xyz_1.shape[1], xyz_1.shape[0])
            # )

            # # Descriptor generation using PointNet or directly using feat_0/feat_1
            # if use_pointnet:
            #     feature_encoder = PointNet().to(device)
            #     feat_0 = feature_encoder(xyz_0, graph_idx_0, None)
            #     feat_1 = feature_encoder(xyz_1, graph_idx_1, None)

            # # Extract edges and edge attributes
            # edges_0, edge_attr_0 = get_edges_batch(graph_idx_0, xyz_0.size(0), xyz_0.size(1))
            # edges_1, edge_attr_1 = get_edges_batch(graph_idx_1, xyz_1.size(0), xyz_1.size(1))

            edges_0 = None
            edge_attr_0 = None
            edges_1 = None
            edge_attr_1 = None
            point_error = None
            feature_loss = None
            # Forward pass through the model
            quaternion, translation, corr_loss, h_src, x_src, h_tgt, x_tgt, gt_labels = model(
                feat_0, xyz_0, edges_0, edge_attr_0, feat_1, xyz_1, edges_1, edge_attr_1, corr, labels
            )
            corr_loss = 1.0
            # Compute additional losses
            point_error, feature_loss = compute_losses(quaternion, translation, h_src, x_src, h_tgt, x_tgt, gt_labels)
            rot_loss, trans_loss = pose_loss(quaternion, translation, gt_pose, delta=1.5)

            # Combine pose and correspondence loss
            loss = rot_loss.mean() + trans_loss.mean() # + beta * corr_loss.mean()

            # Accumulate losses
            running_loss += loss.item()
            running_rot_loss += rot_loss.mean().item()
            running_trans_loss += trans_loss.mean().item()

            # running_corr_loss += corr_loss.mean().item()
            running_point_error += point_error.mean().item()
            # running_feature_consensus_error += feature_loss.mean().item()

        # Compute average losses
        num_batches = len(dataloader)
        avg_loss = running_loss / num_batches
        avg_rot_loss = running_rot_loss / num_batches
        avg_trans_loss = running_trans_loss / num_batches

        # avg_corr_loss = running_corr_loss / num_batches
        avg_point_error = running_point_error / num_batches
        # avg_feature_consensus_error = running_feature_consensus_error / num_batches

        print(
            f"Validation Loss: {avg_loss:.6f} | Pose rot Loss: {avg_rot_loss:.6f} | Pose trans Loss: {avg_trans_loss:.6f} "
            # f"Point error: {avg_point_error:.6f} | Feature error: {avg_feature_consensus_error:.6f}"
        )
        avg_corr_loss = None
    return avg_loss, avg_rot_loss + avg_trans_loss, avg_corr_loss, point_error, feature_loss  #, avg_point_error, avg_feature_consensus_error

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
    scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

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

    if use_pointnet:
        pointnet = PointNet().to(device)
    else:
        pointnet = None

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

            corr = corr.to(device)
            labels = labels.to(device)
            xyz_0, xyz_1 = src_pts.to(device), tar_pts.to(device)
            feat_0, feat_1 = src_features.to(device), tgt_features.to(device)
            gt_pose = gt_pose.to(device)

            k = 12
            graph_idx_0 = knn_graph(xyz_0, k=k, loop=False)
            graph_idx_1 = knn_graph(xyz_1, k=k, loop=False)

            if use_pointnet:
                feat_0 = pointnet(xyz_0, graph_idx_0, None)
                feat_1 = pointnet(xyz_1, graph_idx_1, None)

            edges_0, edge_attr_0 = get_edges_batch(graph_idx_0, xyz_0.size(0), 1)
            edges_1, edge_attr_1 = get_edges_batch(graph_idx_1, xyz_1.size(0), 1)

            quaternion, translation, corr_loss = model(feat_0, xyz_0, edges_0, edge_attr_0, feat_1, xyz_1, edges_1, edge_attr_1, corr, labels)

            for i in range(quaternion.size(0)):
                rot_matrix = quaternion_to_matrix(quaternion[i], device=device)
                trans_vector = translation[i]

                hom_matrix = torch.eye(4, device=device)
                hom_matrix[:3, :3] = rot_matrix
                hom_matrix[:3, 3] = trans_vector

                print(f"Predicted Pose (Batch {batch_idx}, Item {i}):")
                print(hom_matrix)

            pose_losses = pose_loss(quaternion, translation, gt_pose, delta=1.5)
            loss = pose_losses + corr_loss
            total_loss += loss.item()
            total_pose_loss += pose_losses.item()
            total_corr_loss += corr_loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_pose_loss = total_pose_loss / num_batches
    avg_corr_loss = total_corr_loss / num_batches

    print(f"Evaluation completed. Avg Loss: {avg_loss:.6f}, Avg Pose Loss: {avg_pose_loss:.6f}, Avg Corr Loss: {avg_corr_loss:.6f}")
    return avg_loss, avg_pose_loss, avg_corr_loss

def get_args():
    parser = argparse.ArgumentParser(description="Training a Pose Regression Model")
    
    # Add arguments with default values
    parser.add_argument('--base_dir', type=str, default='/home/eavise3d/Downloads/3DMatch_FPFH_Feature', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--num_node', type=int, default=2048, help='Number of nodes in the graph')
    parser.add_argument('--k', type=int, default=12, help='Number of nearest neighbors in KNN graph')
    parser.add_argument('--in_node_nf', type=int, default=32, help='Input feature size for EGNN')
    parser.add_argument('--hidden_node_nf', type=int, default=64, help='Hidden node feature size for EGNN')
    parser.add_argument('--sim_hidden_nf', type=int, default=32, help='Hidden dimension after concatenation in EGNN')
    parser.add_argument('--out_node_nf', type=int, default=32, help='Output node feature size for EGNN')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in EGNN')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "val"], help='Mode to run the model (train/val)')
    parser.add_argument('--lossBeta', type=float, default=1e-2, help='Correspondence loss weights')
    parser.add_argument('--savepath', type=str, default='./checkpoints/model_checkpoint.pth', help='Path to the dataset')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("###edge batch begins ###########")
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
    train_dataset = ThreeDMatchTrainVal(root=base_dir, 
                            split=mode,   
                            descriptor='fpfh',
                            in_dim=6,
                            inlier_threshold=0.10,
                            num_node=2048, 
                            use_mutual=True,
                            downsample=0.03, 
                            augment_axis=1, 
                            augment_rotation=1.0,
                            augment_translation=0.01,
                        )

    val_dataset = ThreeDMatchTrainVal(root=base_dir, 
                            split='val',   
                            descriptor='fpfh',
                            in_dim=6,
                            inlier_threshold=0.10,
                            num_node=2048, 
                            use_mutual=True,
                            downsample=0.03, 
                            augment_axis=1, 
                            augment_rotation=1.0,
                            augment_translation=0.01,
                        )

    test_dataset = ThreeDMatchTest(root=base_dir, 
                            split='test',   
                            descriptor='fpfh',
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
    elif mode == "eval":
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
        avg_loss, avg_pose_loss, avg_corr_loss = evaluate_model(checkpoint_path, cross_attention_model, val_loader, device=dev, use_pointnet=False)