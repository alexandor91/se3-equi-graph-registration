import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

import torch_geometric.transforms as T
from torch_geometric.transforms import SamplePoints, KNNGraph

# from se3_transformer_pytorch import preprocessGcn


def load_h5(file_name):
    f = h5py.File(file_name, mode='r')
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def get_total_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total, 'Trainable': trainable}


def _get_weights(dist, indices):
    num_nodes, k = dist.shape
    print("#######get weights#######")
    print(indices.shape)
    assert num_nodes, k == indices.shape
    assert dist.min() >= 0
    # weight matrix
    sigma2 = np.mean(dist[:, -1])**2
    dist = np.exp(-dist**2 / sigma2)
    i = np.arange(0, num_nodes).repeat(k)
    j = indices.reshape(num_nodes * k)
    v = dist.reshape(num_nodes * k)
    weights = sp.coo_matrix((v, (i, j)), shape=(num_nodes, num_nodes))
    # no self-loop
    weights.setdiag(0)
    # undirected graph
    bigger = weights.T > weights
    weights = weights - weights.multiply(bigger) + weights.T.multiply(bigger)
    return weights


def get_normalize_adj(dist, indices):
    adj = _get_weights(dist, indices)
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def build_graph(coordinates, k=25):
    """
    :param coordinates: positions for 3D point cloud (N * 3)
    :param k: number of nearest neighbors
    :return: adjacency matrix for 3D point cloud
    """
    # from scipy.spatial import cKDTree
    tree = cKDTree(coordinates)
    dist, indices = tree.query(coordinates, k=k)
    return get_normalize_adj(dist, indices)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.conv = nn.Conv1d(
            in_features, out_features, kernel_size=1, bias=bias
        )

    def forward(self, adj, x):
        x = torch.bmm(x, adj)
        x = self.conv(x)
        return x


class GlobalPooling(nn.Module):
    def __init__(self):
        super(GlobalPooling, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        x0 = self.max_pool(x).view(batch_size, -1)
        x1 = self.avg_pool(x).view(batch_size, -1)
        x = torch.cat((x0, x1), dim=-1)
        return x


def main():

    graph = knn_graph(a,k,loop=False)    
    layer = GraphConvolution(in_features=3, out_features=64)
    # print('Parameters:', utils.get_total_parameters(layer))
    x = torch.rand(3,1024)
    adj = build_graph(coordinates=x)#torch.rand(4, 1024, 1024)
    y = layer(adj, x)
    print(y.size())

    pool = GlobalPooling()
    y = pool(y)
    print(y.size())


if __name__ == '__main__':
    main()