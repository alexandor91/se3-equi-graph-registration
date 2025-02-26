
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
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
        super().__init__(aggr='max')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, h: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_j: Tensor, pos_j: Tensor, pos_i: Tensor) -> Tensor:
        # h_j: The features of neighbors as shape [num_edges, in_channels]
        # pos_j: The position of neighbors as shape [num_edges, 3]
        # pos_i: The central node position as shape [num_edges, 3]
        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)

class PointNet(torch.nn.Module):
    def __init__(self, in_num_feature=3, hidden_num_feature=32, output_num_feature=32, device='cuda:0'):
        super().__init__()
        self.hidden_num_feature = hidden_num_feature
        self.conv1 = PointNetLayer(in_num_feature, self.hidden_num_feature)
        self.conv2 = PointNetLayer(self.hidden_num_feature, output_num_feature)
        self.device = device
        
    def forward(self, pos: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        self.to(self.device)
        return h

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
        batch_size, num_edges, _ = x.size()
        x_reshaped = x.view(batch_size * num_edges, self.input_dim, self.input_dim)
        tensor_product = torch.bmm(x_reshaped, x_reshaped.transpose(1, 2))
        tensor_product_flat = tensor_product.view(batch_size, num_edges, -1)
        return self.mlp(tensor_product_flat)

def compute_so3_matrix(x, graph_idx):
    epsilon = 1e-8
    threshold = 1e-6
    idx_i, idx_k = graph_idx
    x_i, x_k = x[idx_i], x[idx_k]
    rel_coord = x_i - x_k
    rel_coord_norm = rel_coord / (rel_coord.norm(dim=-1, keepdim=True) + epsilon)
    cross_prod = torch.cross(x_i, x_k, dim=-1)
    cross_prod_norm = cross_prod / (cross_prod.norm(dim=-1, keepdim=True) + epsilon)
    a_ik, b_ik, c_ik = rel_coord_norm, cross_prod_norm, torch.cross(rel_coord_norm, cross_prod_norm, dim=-1)
    mask = (a_ik.norm(dim=-1) < threshold) | (b_ik.norm(dim=-1) < threshold) | (c_ik.norm(dim=-1) < threshold)
    so3_matrix = torch.stack([a_ik, b_ik, c_ik], dim=-1)
    identity_matrix = torch.eye(3, device=x.device).expand(so3_matrix.shape)
    so3_matrix[mask] = identity_matrix[mask]
    return so3_matrix.view(-1, 9)


def compute_edge_features(coord, edge_index, batch_size):
    """
    Compute edge features (relative coordinates, distances, and dot products) for batched graphs.
    
    Args:
        coord (torch.Tensor): Node coordinates of shape (batch_size, num_nodes, coord_dim).
        edge_index (torch.Tensor): Edge index of shape (2, num_edges_per_batch), shared across batches.
        batch_size (int): Number of graphs in the batch.

    Returns:
        rel_coord (torch.Tensor): Relative coordinates of shape (batch_size, num_edges_per_batch, coord_dim).
        dist (torch.Tensor): Pairwise distances of shape (batch_size, num_edges_per_batch, 1).
        dot_product (torch.Tensor): Dot products of shape (batch_size, num_edges_per_batch, 1).
    """
    row, col = edge_index  # Row and column indices of edge connections
    batch_size, num_nodes, coord_dim = coord.size()

    # Expand edge indices for batched processing
    edge_offset = torch.arange(batch_size, device=coord.device).repeat_interleave(num_nodes) * num_nodes
    row = row + edge_offset
    col = col + edge_offset

    # Flatten coord for easier indexing
    coord_flat = coord.view(-1, coord_dim)  # Shape: (batch_size * num_nodes, coord_dim)

    # Compute relative coordinates, distances, and dot products
    rel_coord = coord_flat[row] - coord_flat[col]  # Relative positions
    dist = torch.norm(rel_coord, dim=-1, keepdim=True)  # ∥xi - xj∥
    dot_product = (coord_flat[row] * coord_flat[col]).sum(dim=-1, keepdim=True)  # xi ⋅ xj

    # Reshape outputs back into batched dimensions
    num_edges_per_batch = row.size(0) // batch_size
    rel_coord = rel_coord.view(batch_size, num_edges_per_batch, coord_dim)
    dist = dist.view(batch_size, num_edges_per_batch, 1)
    dot_product = dot_product.view(batch_size, num_edges_per_batch, 1)

    return rel_coord, dist, dot_product


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

    def edge_model(self, source, target, radial, edge_attr, coord, edge_index, batch_size):
        rel_coord, dist, dot_product = compute_edge_features(coord, edge_index, batch_size)

        so3_flat = compute_so3_matrix(coord, edge_index, batch_size)  # From your code
        # Combine features
        features = [source, target, radial, dist, dot_product, so3_flat]
        if edge_attr is not None:
            features.append(edge_attr)
        out = torch.cat(features, dim=-1)

        # Multi-head MLP
        head_outputs = [mlp(out) for mlp in self.edge_mlps]
        combined = torch.cat(head_outputs, dim=-1)

        # Layer normalization
        out = self.layer_norm(combined)
        return out

    def node_model(self, x, edge_index, edge_attr, batch_size):
        row = edge_index[0]
        num_nodes = x.size(1)
        agg = unsorted_segment_sum(edge_attr, row, num_segments=batch_size * num_nodes)
        agg = agg.view(batch_size, num_nodes, -1)  # Reshape to match batch size
        out = torch.cat([x, agg], dim=-1)
        out = self.node_mlp(out)
        if self.residual:
            out = x + out
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, batch_size):
        row = edge_index[0]
        num_nodes = coord.size(1)
        trans = coord_diff * self.coord_mlp(edge_feat)
        agg = unsorted_segment_sum(trans, row, num_segments=batch_size * num_nodes)
        agg = agg.view(batch_size, num_nodes, -1)  # Reshape to match batch size
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord, batch_size):
        row, col = edge_index
        coord_diff = coord[:, row] - coord[:, col]
        radial = torch.sum(coord_diff**2, dim=-1, keepdim=True)
        if self.normalize:
            norm = torch.sqrt(radial).detach() + 1e-6
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None):
        batch_size, num_nodes, _ = h.size()
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord, batch_size)

        edge_feat = self.edge_model(h[:, row], h[:, col], radial, edge_attr, coord, edge_index, batch_size)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, batch_size)
        h = self.node_model(h, edge_index, edge_feat, batch_size)

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
        
        for i in range(n_layers):
            self.add_module(
                f"gcl_{i}", 
                E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                      act_fn=act_fn, residual=residual, attention=attention,
                      normalize=normalize, tanh=tanh)
            )

    def forward(self, h, x, edges, edge_attr, batch_size):
        h = self.embedding_in(h)
        for i in range(self.n_layers):
            h, x, _ = self._modules[f"gcl_{i}"](h, edges, x, edge_attr=edge_attr)
        
        # Reshape for batching, assuming h has shape [batch_size * num_nodes, hidden_dim]
        h = self.embedding_out(h)
        return h.view(batch_size, -1, h.size(-1)), x.view(batch_size, -1, x.size(-1))

def unsorted_segment_sum(data, segment_ids, num_segments):
    # Handling the batch-specific version of the segment sum
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
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
    return [rows, cols]

def get_edges_from_idx(graph_idx):
    src, dst = graph_idx[0], graph_idx[1]
    return [src, dst]

def get_edges_batch(graph_idx, n_nodes, batch_size):
    edges = get_edges_from_idx(graph_idx)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1).to(graph_idx.device)
    
    if batch_size == 1:
        return edges, edge_attr
    else:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        
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

def rotation_matrix_to_quaternion(rotation_matrices):
    """
    Convert 3x3 rotation matrices to quaternions (supports batch size).
    Args:
        rotation_matrices (Tensor): Batch of rotation matrices of shape [B, 3, 3].
    Returns:
        Tensor: Batch of quaternions of shape [B, 4].
    """
    assert rotation_matrices.dim() == 3 and rotation_matrices.shape[1:] == (3, 3), \
        "Input must have shape [B, 3, 3]"
    
    batch_size = rotation_matrices.shape[0]
    quaternions = torch.zeros((batch_size, 4), device=rotation_matrices.device)
    m = rotation_matrices

    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]  # Trace of the matrix

    # Case 1: t > 0
    mask_t_positive = t > 0
    s = torch.sqrt(1.0 + t[mask_t_positive]) * 2
    quaternions[mask_t_positive, 0] = 0.25 * s
    quaternions[mask_t_positive, 1] = (m[mask_t_positive, 2, 1] - m[mask_t_positive, 1, 2]) / s
    quaternions[mask_t_positive, 2] = (m[mask_t_positive, 0, 2] - m[mask_t_positive, 2, 0]) / s
    quaternions[mask_t_positive, 3] = (m[mask_t_positive, 1, 0] - m[mask_t_positive, 0, 1]) / s

    # Case 2: m[0, 0] is largest
    mask_m00_largest = ~mask_t_positive & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    s = torch.sqrt(1.0 + m[mask_m00_largest, 0, 0] - m[mask_m00_largest, 1, 1] - m[mask_m00_largest, 2, 2]) * 2
    quaternions[mask_m00_largest, 0] = (m[mask_m00_largest, 2, 1] - m[mask_m00_largest, 1, 2]) / s
    quaternions[mask_m00_largest, 1] = 0.25 * s
    quaternions[mask_m00_largest, 2] = (m[mask_m00_largest, 0, 1] + m[mask_m00_largest, 1, 0]) / s
    quaternions[mask_m00_largest, 3] = (m[mask_m00_largest, 0, 2] + m[mask_m00_largest, 2, 0]) / s

    # Case 3: m[1, 1] is largest
    mask_m11_largest = ~mask_t_positive & ~mask_m00_largest & (m[:, 1, 1] > m[:, 2, 2])
    s = torch.sqrt(1.0 + m[mask_m11_largest, 1, 1] - m[mask_m11_largest, 0, 0] - m[mask_m11_largest, 2, 2]) * 2
    quaternions[mask_m11_largest, 0] = (m[mask_m11_largest, 0, 2] - m[mask_m11_largest, 2, 0]) / s
    quaternions[mask_m11_largest, 1] = (m[mask_m11_largest, 0, 1] + m[mask_m11_largest, 1, 0]) / s
    quaternions[mask_m11_largest, 2] = 0.25 * s
    quaternions[mask_m11_largest, 3] = (m[mask_m11_largest, 1, 2] + m[mask_m11_largest, 2, 1]) / s

    # Case 4: m[2, 2] is largest
    mask_m22_largest = ~mask_t_positive & ~mask_m00_largest & ~mask_m11_largest
    s = torch.sqrt(1.0 + m[mask_m22_largest, 2, 2] - m[mask_m22_largest, 0, 0] - m[mask_m22_largest, 1, 1]) * 2
    quaternions[mask_m22_largest, 0] = (m[mask_m22_largest, 1, 0] - m[mask_m22_largest, 0, 1]) / s
    quaternions[mask_m22_largest, 1] = (m[mask_m22_largest, 0, 2] + m[mask_m22_largest, 2, 0]) / s
    quaternions[mask_m22_largest, 2] = (m[mask_m22_largest, 1, 2] + m[mask_m22_largest, 2, 1]) / s
    quaternions[mask_m22_largest, 3] = 0.25 * s

    return quaternions

def quaternion_to_matrix(quaternions):
    """
    Converts a batch of quaternions into 3x3 rotation matrices.
    Args:
        quaternions (Tensor): Batch of quaternions of shape [B, 4].
    Returns:
        Tensor: Batch of 3x3 rotation matrices of shape [B, 3, 3].
    """
    assert quaternions.dim() == 2 and quaternions.shape[1] == 4, \
        "Input must have shape [B, 4]"
    
    q = quaternions
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Compute the elements of the rotation matrix
    r00 = 1 - 2 * (y ** 2 + z ** 2)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)

    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x ** 2 + z ** 2)
    r12 = 2 * (y * z - x * w)

    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x ** 2 + y ** 2)

    # Stack the rotation matrix for each quaternion
    rotation_matrices = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=-2)

    return rotation_matrices


class CrossAttentionPoseRegression(nn.Module):
    def __init__(self, egnn: nn.Module, num_nodes: int = 2048, hidden_nf: int = 35, device='cuda:0'):
        super(CrossAttentionPoseRegression, self).__init__()
        self.egnn = egnn  # The shared EGNN network
        self.hidden_nf = hidden_nf  # Hidden feature dimension (35)
        self.num_nodes = num_nodes
        self.device = device  # Device (CPU or GPU)

        # MLP layers to compress the input from 2048 points to 128 points
        self.mlp_compress = nn.Sequential(
            nn.Linear(num_nodes, num_nodes//4),  # Compress to 128 points
            nn.ReLU(),
            nn.Linear(num_nodes//4, 128)  # Keep 128 points compressed
        )

        # MLP for pose regression (quaternion + translation)
        self.mlp_pose = nn.Sequential(
            nn.Linear(128 * 2 * self.hidden_nf, 256),  # Concatenate source & target and compress
            nn.ReLU(),
            nn.Linear(256, 128),  # Intermediate layer
            nn.ReLU(),
            nn.Linear(128, 64),  # Intermediate layer
            nn.ReLU(),
            nn.Linear(64, 7)  # Output quaternion (4) + translation (3)
        )

    def _preprocess_tensor(self, tensor, device):
        """
        Preprocess tensor to ensure it's on the correct device and handle potential batching.
        
        Args:
            tensor (torch.Tensor): Input tensor
            device (str): Target device
        
        Returns:
            torch.Tensor: Processed tensor
        """
        # Ensure tensor is on the correct device
        tensor = tensor.to(device)
        
        # If tensor is 3D (batch, nodes, features), squeeze if batch size is 1
        if tensor.dim() == 3 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        
        return tensor

    def _preprocess_edges(self, edges, device):
        """
        Preprocess edge indices to ensure they're on the correct device.
        
        Args:
            edges (torch.Tensor or list): Edge indices
            device (str): Target device
        
        Returns:
            torch.Tensor or list: Processed edge indices
        """
        if edges is None:
            return None
        
        if isinstance(edges, list):
            return [edge.to(device) for edge in edges]
        else:
            return edges.to(device)

    def forward(self, h_src, x_src, edges_src, edge_attr_src, 
                h_tgt, x_tgt, edges_tgt, edge_attr_tgt, 
                corr, labels):
        """
        Forward pass with support for batched inputs.
        
        Expected input shapes:
        - h_src, x_src: [batch, num_nodes, features] or [num_nodes, features]
        - edges_src: list of tensors or tensor of edge indices
        - corr: [batch, num_correspondences, 2] or [num_correspondences, 2]
        - labels: [batch, num_correspondences] or [num_correspondences]
        
        Returns:
        - quaternion: [batch, 4]
        - translation: [batch, 3]
        - total_corr_loss: scalar loss
        """
        # Determine if input is batched
        is_batched = h_src.dim() == 3
        batch_size = h_src.size(0) if is_batched else 1

        # Move everything to the same device (GPU/CPU)
        h_src = self._preprocess_tensor(h_src, self.device)
        x_src = self._preprocess_tensor(x_src, self.device)
        edges_src = self._preprocess_edges(edges_src, self.device)
        edge_attr_src = edge_attr_src.to(self.device) if edge_attr_src is not None else None

        h_tgt = self._preprocess_tensor(h_tgt, self.device)
        x_tgt = self._preprocess_tensor(x_tgt, self.device)
        edges_tgt = self._preprocess_edges(edges_tgt, self.device)
        edge_attr_tgt = edge_attr_tgt.to(self.device) if edge_attr_tgt is not None else None

        # Preprocess corr and labels
        corr = corr.to(self.device)
        labels = labels.to(self.device)

        # Squeeze labels if batched and single batch
        if labels.dim() == 2 and labels.size(0) == 1:
            labels = labels.squeeze(0)

        # Process batched or single inputs
        if is_batched:
            # Initialize lists to store results
            quaternions = []
            translations = []
            corr_losses = []

            # Process each batch item
            for i in range(batch_size):
                # Extract batch-specific data
                h_src_i = h_src[i]
                x_src_i = x_src[i]
                h_tgt_i = h_tgt[i]
                x_tgt_i = x_tgt[i]
                
                # Get batch-specific correspondence and labels
                batch_mask = corr[:, 0] == i
                corr_i = corr[batch_mask, 1:]  # Remove batch index
                labels_i = labels[batch_mask]

                # Process this batch item
                quat_i, trans_i, corr_loss_i = self._process_single_item(
                    h_src_i, x_src_i, edges_src, edge_attr_src,
                    h_tgt_i, x_tgt_i, edges_tgt, edge_attr_tgt,
                    corr_i, labels_i
                )

                quaternions.append(quat_i)
                translations.append(trans_i)
                corr_losses.append(corr_loss_i)

            # Stack results
            quaternions = torch.stack(quaternions)
            translations = torch.stack(translations)
            total_corr_loss = torch.mean(torch.stack(corr_losses))

            return quaternions, translations, total_corr_loss
        else:
            # Process single item
            return self._process_single_item(
                h_src, x_src, edges_src, edge_attr_src,
                h_tgt, x_tgt, edges_tgt, edge_attr_tgt,
                corr, labels
            )

    def _process_single_item(self, h_src, x_src, edges_src, edge_attr_src,
                             h_tgt, x_tgt, edges_tgt, edge_attr_tgt,
                             corr, labels):
        """
        Process a single item (used internally by forward method).
        
        Args:
            h_src, x_src, h_tgt, x_tgt: [num_nodes, features]
            edges_src, edges_tgt: edge indices
            corr: [num_correspondences, 2]
            labels: [num_correspondences]
        
        Returns:
            quaternion, translation, total_corr_loss
        """
        # Process source and target point clouds with the EGNN
        h_src, x_src = self.egnn(h_src, x_src, edges_src, edge_attr_src)  # Shape: [2048, hidden_nf]
        h_tgt, x_tgt = self.egnn(h_tgt, x_tgt, edges_tgt, edge_attr_tgt)  # Shape: [2048, hidden_nf]

        # Concatenate node features with coordinates
        h_src = torch.cat([h_src, x_src], dim=-1)  # Shape: [2048, 35]
        h_tgt = torch.cat([h_tgt, x_tgt], dim=-1)  # Shape: [2048, 35]

        # Normalize features for Hadamard product (L2 normalization)
        h_src_norm = F.normalize(h_src, p=2, dim=-1)
        h_tgt_norm = F.normalize(h_tgt, p=2, dim=-1)

        # Compute similarity matrix
        large_sim_matrix = torch.mm(h_src_norm, h_tgt_norm.t())  # Shape: [num_nodes, num_nodes]
        
        # Get correspondence indices
        corr_indices = corr[:, 0].long(), corr[:, 1].long()

        # Get similarity at correspondence indices
        corr_similarity = large_sim_matrix[corr_indices]

        # Correspondence loss computation
        corr_loss = F.mse_loss(corr_similarity, labels*torch.ones_like(corr_similarity).to(self.device))

        # Compress features to 128 dimensions
        compressed_h_src = self.mlp_compress(h_src.transpose(0, 1)).transpose(0, 1)
        compressed_h_tgt = self.mlp_compress(h_tgt.transpose(0, 1)).transpose(0, 1)

        compressed_h_src_norm = F.normalize(compressed_h_src, p=2, dim=-1)
        compressed_h_tgt_norm = F.normalize(compressed_h_tgt, p=2, dim=-1)

        # Compute similarity matrix (dot product)
        sim_matrix = torch.mm(compressed_h_src_norm, compressed_h_tgt_norm.t())  # Shape: [128, 128]

        # Compute rank loss
        u, s, v = torch.svd(sim_matrix)
        rank_loss = F.mse_loss(s[:128], torch.ones(128).to(self.device))

        # Total correspondence loss
        total_corr_loss = corr_loss + rank_loss

        # Weigh the source and target descriptors using the similarity matrix
        weighted_h_src = torch.mm(sim_matrix.transpose(0, 1), compressed_h_src)  # Shape: [128, 35]
        weighted_h_tgt = torch.mm(sim_matrix, compressed_h_tgt)  # Shape: [128, 35]

        # Concatenate the source and target weighted features
        combined_features = torch.cat([weighted_h_src, weighted_h_tgt], dim=-1)  # Shape: [128, 70]

        # Flatten the combined features for pose regression
        combined_features_flat = combined_features.view(-1)  # Shape: [128 * 70]

        # Regress the pose (quaternion + translation)
        pose = self.mlp_pose(combined_features_flat)  # Shape: [7]

        # Separate quaternion and translation
        quaternion = pose[:4]  # Shape: [4]
        translation = pose[4:]  # Shape: [3]

        return quaternion, translation, total_corr_loss
