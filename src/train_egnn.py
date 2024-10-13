
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
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        # h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        # h = h.relu()
        
        # h = self.conv3(h=h, pos=pos, edge_index=edge_index)
        # h = h.relu()
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
    E(n) Equivariant Convolutional Layer, basic moduel to buil up the eqnn network model
    re
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
        self.device =device

        # Edge MLP, with 9 additional dimensions for SO(3) flattened feature
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d + 9, hidden_nf), ########2*input node feature, edge feature, edge attribute dimension features
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

    def edge_model(self, source, target, radial, edge_attr, coord, edge_index):
        # Compute the SO(3) matrix features (N x 9)
        so3_flat = compute_so3_matrix(coord, edge_index)
        # so3_output = self.so3_tensor_product(so3_flat)    # Apply SO(3) tensor product layer


        if edge_attr is None:  # Unused for edge attributes.
            out = torch.cat([source, target, radial, so3_flat], dim=1)
        else:
            # Concatenate source, target, radial, edge_attr, and SO(3) flattened features
            out = torch.cat([source, target, radial, edge_attr, so3_flat], dim=1)

        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

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
        if edge_feat is not None:
            trans = coord_diff * self.coord_mlp(edge_feat)
        trans = coord_diff
        # print(self.coord_mlp(edge_feat))
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
        self.to(self.device)
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)


        # Edge model with SO(3) features concatenated
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, coord, edge_index)

        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # print(coord.shape)
        # print(h.shape)
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
    Convert a 3x3 rotation matrix to a quaternion (no batch size).
    Args:
        rotation_matrix (Tensor): Rotation matrix of shape [3x3].
    Returns:
        Tensor: Quaternion tensor [4].
    """
    quaternion = torch.zeros(4, device=rotation_matrix.device)
    
    m = rotation_matrix
    t = m[0, 0] + m[1, 1] + m[2, 2]
    
    if t > 0:
        s = torch.sqrt(1.0 + t) * 2
        quaternion[0] = 0.25 * s
        quaternion[1] = (m[2, 1] - m[1, 2]) / s
        quaternion[2] = (m[0, 2] - m[2, 0]) / s
        quaternion[3] = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        quaternion[0] = (m[2, 1] - m[1, 2]) / s
        quaternion[1] = 0.25 * s
        quaternion[2] = (m[0, 1] + m[1, 0]) / s
        quaternion[3] = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        quaternion[0] = (m[0, 2] - m[2, 0]) / s
        quaternion[1] = (m[0, 1] + m[1, 0]) / s
        quaternion[2] = 0.25 * s
        quaternion[3] = (m[1, 2] + m[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        quaternion[0] = (m[1, 0] - m[0, 1]) / s
        quaternion[1] = (m[0, 2] + m[2, 0]) / s
        quaternion[2] = (m[1, 2] + m[2, 1]) / s
        quaternion[3] = 0.25 * s

    return quaternion

class CrossAttentionPoseRegression(nn.Module):
    def __init__(self, egnn: EGNN, num_nodes: int = 2048, hidden_nf: int = 35, device='cuda:0'):
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

    def forward(self, h_src, x_src, edges_src, edge_attr_src, h_tgt, x_tgt, edges_tgt, edge_attr_tgt, corr, labels):
        # Move everything to the same device (GPU/CPU)
        h_src = h_src.to(self.device)
        x_src = x_src.to(self.device)

        # Ensure edge indices are tensors and move them to the correct device
        if isinstance(edges_src, list):
            edges_src = [edge.to(self.device) for edge in edges_src]
        else:
            edges_src = edges_src.to(self.device)

        if edge_attr_src is not None:
            edge_attr_src = edge_attr_src.to(self.device)

        h_tgt = h_tgt.to(self.device)
        x_tgt = x_tgt.to(self.device)

        # Ensure edge indices for the target are tensors and move them to the correct device
        if isinstance(edges_tgt, list):
            edges_tgt = [edge.to(self.device) for edge in edges_tgt]
        else:
            edges_tgt = edges_tgt.to(self.device)

        if edge_attr_tgt is not None:
            edge_attr_tgt = edge_attr_tgt.to(self.device)

        if labels.dim() == 2 and labels.size(0) == 1:
            labels = labels.squeeze(0)

        # Process source and target point clouds with the EGNN
        h_src, x_src = self.egnn(h_src, x_src, edges_src, edge_attr_src)  # Shape: [2048, hidden_nf]
        h_tgt, x_tgt = self.egnn(h_tgt, x_tgt, edges_tgt, edge_attr_tgt)  # Shape: [2048, hidden_nf]

        # Concatenate node features with coordinates
        h_src = torch.cat([h_src, x_src], dim=-1)  # Shape: [2048, 35]
        h_tgt = torch.cat([h_tgt, x_tgt], dim=-1)  # Shape: [2048, 35]

        # Normalize features for Hadamard product (L2 normalization)
        h_src_norm = F.normalize(h_src, p=2, dim=-1)
        h_tgt_norm = F.normalize(h_tgt, p=2, dim=-1)

        large_sim_matrix = torch.mm(h_src_norm, h_tgt_norm.t())  # Shape: [num_nodes, num_nodes]
        # print(large_sim_matrix.shape)
        # print(corr.shape)
        # print(labels.shape)
        corr_indices = corr[:, 0].long(), corr[:, 1].long()  # Convert to LongTensor for indexing        
        # print(corr[:, 0].long())

        # Get similarity at correspondence indices
        corr_similarity = large_sim_matrix[corr_indices]  # Get correspondence similarity values
        # print(corr_similarity.shape)

        # Correspondence loss computation
        corr_loss = F.mse_loss(corr_similarity, labels*torch.ones_like(corr_similarity).cuda())  # Correspondence should be close to 1

        # Compress features to 128 dimensions
        compressed_h_src = self.mlp_compress(h_src.transpose(0, 1)).transpose(0, 1)
        compressed_h_tgt = self.mlp_compress(h_tgt.transpose(0, 1)).transpose(0, 1)

        compressed_h_src_norm = F.normalize(self.mlp_compress(h_src.transpose(0, 1)).transpose(0, 1), p=2, dim=-1)
        compressed_h_tgt_norm = F.normalize(self.mlp_compress(h_tgt.transpose(0, 1)).transpose(0, 1), p=2, dim=-1)

        # Compute similarity matrix (dot product)
        sim_matrix = torch.mm(compressed_h_src_norm, compressed_h_tgt_norm.t())  # Shape: [128, 128]

        # Further compute rank loss for compressed similarity matrix
        u, s, v = torch.svd(sim_matrix)
        rank_loss = F.mse_loss(s[:128], torch.ones(128).cuda())  # Rank close to 128

        # Correspondence loss: Combine similarity-based loss and rank loss
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

        # The pose consists of a quaternion (4D) and a translation (3D)
        quaternion = pose[:4]  # Shape: [4]
        translation = pose[4:]  # Shape: [3]
        return quaternion, translation, total_corr_loss
    

def pose_loss(pred_quaternion, pred_translation, gt_pose, delta=1.0):
    """
    Compute the loss between the predicted pose (quaternion + translation) and the ground truth pose matrix.
    
    Arguments:
    - pred_quaternion: Predicted quaternion of shape [4]
    - pred_translation: Predicted translation of shape [3]
    - gt_pose: Ground truth pose as a 4x4 matrix (rotation matrix + translation)
    
    Returns:
    - Total loss combining quaternion and translation loss.
    """
    
    # Extract ground truth translation (3D) and rotation (3x3) from the 4x4 gt_pose matrix
    gt_translation = gt_pose[:3, 3]    # Translation vector [3]
    gt_rotation = gt_pose[:3, :3]      # Rotation matrix [3x3]

    # Convert ground truth rotation matrix to quaternion
    gt_quaternion = rotation_matrix_to_quaternion(gt_rotation)  # Convert [3x3] to [4]
    gt_quaternion = F.normalize(gt_quaternion, p=2, dim=-1)
    # Normalize the predicted quaternion
    pred_quaternion = F.normalize(pred_quaternion, p=2, dim=-1)

    # Huber loss for quaternion and translation
    huber_loss = nn.HuberLoss(delta=delta)

    quaternion_loss = huber_loss(pred_quaternion, gt_quaternion)
    translation_loss = huber_loss(pred_translation, gt_translation)

    # Return the combined loss
    return quaternion_loss + translation_loss

# Function to train for one epoch
def train_one_epoch(model, dataloader, optimizer, device, use_pointnet=False, log_interval=10, beta=0.1):
    model.train()
    running_loss = 0.0

    for batch_idx, (corr, labels, src_pts, tar_pts, src_features, tgt_features, gt_pose) in enumerate(dataloader):
        # Check if any returned data is None, and skip if so
        if corr is None or labels is None or src_pts is None or tar_pts is None or src_features is None or tgt_features is None or gt_pose is None:
            print(f"Skipping batch {batch_idx} due to missing data.")
            continue  # Skip this batch

        # Move the data to the specified device (GPU/CPU)
        corr = corr.to(device)
        labels = labels.to(device)
        xyz_0, xyz_1 = src_pts.to(device), tar_pts.to(device)
        feat_0, feat_1 = src_features.to(device), tgt_features.to(device)
        gt_pose = gt_pose.to(device)

        # print("################@@@@@@@@@@@@@@@@######################")
        # print(xyz_0.shape)
        # print(xyz_1.shape)
        # print(feat_0.shape)
        # print(feat_1.shape)
 
        # Initialize KNN graphs for source and target point clouds
        k = 12
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

        # # Forward pass through the model
        quaternion, translation, corr_loss = model(feat_0, xyz_0, edges_0, edge_attr_0, feat_1, xyz_1, edges_1, edge_attr_1, corr, labels)
        #corr_loss

        # # Compute pose loss
        pose_losses = pose_loss(quaternion, translation, gt_pose, delta=1.5)

        # # Combine pose and correspondence loss
        loss = pose_losses + beta*corr_loss

        # # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # Print loss for every log_interval batches
        if (batch_idx+1) % log_interval == 0:
            print(f'Iteration {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}')

    return running_loss / len(dataloader)

# Function to validate the model on the validation set
def validate(model, dataloader, device, use_pointnet=False):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0  # Track overall loss
    running_corr_loss = 0.0  # Track correspondence rank loss
    running_pose_loss = 0.0  # Track pose loss

    with torch.no_grad():  # No need to track gradients during validation
        for batch_idx, (corr, labels, src_pts, tar_pts, src_features, tgt_features, gt_pose) in enumerate(dataloader):
            # Check if any returned data is None, and skip if so
            if corr is None or labels is None or src_pts is None or tar_pts is None or src_features is None or tgt_features is None or gt_pose is None:
                print(f"Skipping batch {batch_idx} due to missing data.")
                continue  # Skip this batch

            # Move the data to the specified device (GPU/CPU)
            corr = corr.to(device)
            labels = labels.to(device)
            xyz_0, xyz_1 = src_pts.to(device), tar_pts.to(device)
            feat_0, feat_1 = src_features.to(device), tgt_features.to(device)
            gt_pose = gt_pose.to(device)

            # print("################@@@@@@@@@@@@@@@@######################")
            # print(xyz_0.shape)
            # print(xyz_1.shape)
            # print(feat_0.shape)
            # print(feat_1.shape)
    
            # Initialize KNN graphs for source and target point clouds
            k = 12
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
            quaternion, translation, corr_loss = model(feat_0, xyz_0, edges_0, edge_attr_0, feat_1, xyz_1, edges_1, edge_attr_1, corr, labels)
            #corr_loss

            # # Compute pose loss
            pose_losses = pose_loss(quaternion, translation, gt_pose, delta=1.5)

            # # Combine pose and correspondence loss
            loss = pose_losses + beta*corr_loss

            # Accumulate losses
            running_loss += loss.item()
            running_pose_loss += pose_losses.item()
            running_corr_loss += corr_loss.item()

        # Return the average losses over the entire validation dataset
        avg_loss = running_loss / len(dataloader)
        avg_pose_loss = running_pose_loss / len(dataloader)
        avg_corr_loss = running_corr_loss / len(dataloader)

    print(f'Validation Loss: {avg_loss:.4f} | Pose Loss: {avg_pose_loss:.4f} | Correspondence Loss: {avg_corr_loss:.4f}')

    return avg_loss, avg_pose_loss, avg_corr_loss

# Save the checkpoint
def save_checkpoint(epoch, pointnet, egnn, cross_attention, optimizer, save_dir="./checkpoints", use_pointnet=False):
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

# Main training loop
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, use_pointnet=False, log_interval=10, beta=0.1, save_path="./checkpoints"):
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
        train_loss = train_one_epoch(model, train_loader, optimizer, device, use_pointnet, log_interval, beta)

        # Validate every few epochs (e.g., every 5 epochs)
        if (epoch + 1) % 1 == 0:
            val_loss, val_pose_loss, val_corr_loss = validate(model, val_loader, device, use_pointnet)
            print(val_loss)
            print(f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        else:
            print(f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_loss:.4f}')

        # Save checkpoint every 16 epochs
        if (epoch + 1) % 16 == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = "./checkpoints"
            save_checkpoint(epoch + 1, pointnet, egnn, cross_attention, optimizer, checkpoint_path, use_pointnet=use_pointnet)

def get_args():
    parser = argparse.ArgumentParser(description="Training a Pose Regression Model")
    
    # Add arguments with default values
    parser.add_argument('--base_dir', type=str, default='/home/eavise3d/3DMatch_FCGF_Feature_32_transform', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--num_node', type=int, default=2048, help='Number of nodes in the graph')
    parser.add_argument('--k', type=int, default=12, help='Number of nearest neighbors in KNN graph')
    parser.add_argument('--in_node_nf', type=int, default=32, help='Input feature size for EGNN')
    parser.add_argument('--hidden_node_nf', type=int, default=64, help='Hidden node feature size for EGNN')
    parser.add_argument('--sim_hidden_nf', type=int, default=35, help='Hidden dimension after concatenation in EGNN')
    parser.add_argument('--out_node_nf', type=int, default=32, help='Output node feature size for EGNN')
    parser.add_argument('--n_layers', type=int, default=5, help='Number of layers in EGNN')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "val"], help='Mode to run the model (train/val)')
    parser.add_argument('--lossBeta', type=float, default=0.05, help='Correspondence loss weights')
    parser.add_argument('--savepath', type=str, default='./checkpoints/model_checkpoint.pth', help='Path to the dataset')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("###edge batch begins 2###########")
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

    mode = "train"
    train_dataset = ThreeDMatchTrainVal(root=base_dir, 
                            split=mode,   
                            descriptor='fcgf',
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
                            descriptor='fcgf',
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

    # # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    egnn = EGNN(in_node_nf=in_node_nf, hidden_nf=hidden_node_nf, out_node_nf=out_node_nf, in_edge_nf=1, n_layers=n_layers)
    egnn.to(dev)
    # egnn(h, x, edges1, edge_attr1)
    # print("###edge batch ends###########")
    cross_attention_model = CrossAttentionPoseRegression(egnn=egnn, num_nodes=num_node, hidden_nf=sim_hidden_nf, device=dev).to(dev)
    cross_attention_model.to(dev)

    # cross_attention_model(h, x, graph_idx1, edge_attr1, h, x, graph_idx1, edge_attr1)

    train_model(cross_attention_model, train_loader, val_loader, num_epochs=num_epochs, \
                learning_rate=learning_rate, device=dev, use_pointnet=False, log_interval=10, beta=0.1, save_path=savepath)
