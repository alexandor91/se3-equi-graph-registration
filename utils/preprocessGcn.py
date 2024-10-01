# import argparse
import h5py

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


# def import_class(name):
#     try:
#         components = name.split('.')
#         module = __import__(components[0])
#         for c in components[1:]:
#             module = getattr(module, c)
#     except AttributeError:
#         module = None
#     return module


def load_h5(file_name):
    f = h5py.File(file_name, mode='r')
    data = f['data'][:]
    label = f['label'][:]
    return data, label


# def str2bool(v):
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')


def get_total_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total, 'Trainable': trainable}


def _get_weights(dist, indices):
    num_nodes, k = dist.shape
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


def _get_normalize_adj(dist, indices):
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
    return _get_normalize_adj(dist, indices)