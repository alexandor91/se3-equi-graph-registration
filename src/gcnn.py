import torch
from torch_geometric.data import Data
from torch_geometric.nn import DynamicEdgeConv
from torch_geometric.nn import knn_graph
import os
import numpy as np

# Load your point cloud data into PyTorch Geometric Data object
# Assume your data is a NumPy array with shape (num_points, 3)
base_dir = "/home/eavise3d/Downloads/visualization/t-SNE-viz-spintnet/test_data"
src_filename = "cloud_bin_26_kpts.npy"
tar_filename = "rotate_cloud_bin_26_kpts.npy"

src_points = np.load(os.path.join(base_dir, src_filename))
tar_points = np.load(os.path.join(base_dir, tar_filename))


# Ensure the point cloud has more than 1024 points for downsampling
if src_points.shape[0] > 1024:
    src_points = src_points[np.random.choice(src_points.shape[0], 1024, replace=False)]

# Convert numpy array to tensor
src_points_tensor = torch.from_numpy(src_points).float()

# Create a dummy batch tensor for a single point cloud
num_points = src_points_tensor.size(0)
batch_tensor = torch.zeros(num_points, dtype=torch.long)

# Create edge index using knn_graph
# edge_index = knn_graph(src_points_tensor, k=12, batch_x=batch_x, batch_y=batch_x)
edge_index = knn_graph(src_points_tensor, k=12, batch=batch_tensor)
# Create PyTorch Geometric Data object
data = Data(x=src_points_tensor, edge_index=edge_index)

# Ensure the edge index and node features have consistent dimensions
assert edge_index.size(0) == 2, f"Expected edge_index of size 2, got {edge_index.size(0)}"
assert src_points_tensor.size(0) == edge_index.max().item() + 1, f"Number of nodes {src_points_tensor.size(0)} does not match max index in edge_index {edge_index.max().item() + 1}"

# Create PyTorch Geometric data object
data = Data(x=src_points_tensor, edge_index=edge_index)

# Define MLP for DynamicEdgeConv (assuming it is a simple sequential model)
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])

# Create the DynamicEdgeConv model
model = DynamicEdgeConv(MLP([3, 32, 32, 32, 32]), k=10)

# Load pre-trained model weights if they exist (this is a placeholder, adjust as needed)
# model.load_state_dict(torch.load('path_to_pretrained_weights.pth'))  # Uncomment and provide path

# Get node features with torch.no_grad():
with torch.no_grad():
    node_features = model(data.x, data.edge_index)

print(node_features)

# Visualize node features using your preferred library (e.g., Open3D, Matplotlib)
# Example with Matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

node_features_np = node_features.cpu().numpy()
node_features_2d = TSNE(n_components=2).fit_transform(node_features_np)

plt.scatter(node_features_2d[:, 0], node_features_2d[:, 1])
plt.title("t-SNE Visualization of Node Features")
plt.show()
# Ensure the point cloud has more than 1024 points for downsampling
# if src_points.shape[0] > 1024:
#     src_points = src_points[np.random.choice(src_points.shape[0], 1024, replace=False)]

# # Convert numpy array to tensor
# src_points_tensor = torch.from_numpy(src_points).float()

# # Create edge_index using knn_graph
# edge_index = knn_graph(src_points_tensor, k=12)

# # Ensure the edge index and node features have consistent dimensions
# assert edge_index.size(0) == 2, f"Expected edge_index of size 2, got {edge_index.size(0)}"
# assert src_points_tensor.size(0) == edge_index.max().item() + 1, f"Number of nodes {src_points_tensor.size(0)} does not match max index in edge_index {edge_index.max().item() + 1}"

# # Create PyTorch Geometric data object
# data = Data(x=src_points_tensor, edge_index=edge_index)

# # Define MLP for DynamicEdgeConv (assuming it is a simple sequential model)
# from torch.nn import Sequential as Seq, Linear as Lin, ReLU

# def MLP(channels):
#     return Seq(*[
#         Seq(Lin(channels[i - 1], channels[i]), ReLU())
#         for i in range(1, len(channels))
#     ])

# # Create the DynamicEdgeConv model
# model = DynamicEdgeConv(MLP([3, 32, 32, 32, 32]), k=10)

# # Load pre-trained model weights if they exist (this is a placeholder, adjust as needed)
# # model.load_state_dict(torch.load('path_to_pretrained_weights.pth'))  # Uncomment and provide path

# # Get node features with torch.no_grad():
# with torch.no_grad():
#     node_features = model(data.x, data.edge_index)

# print(node_features)

# # Visualize node features using your preferred library (e.g., Open3D, Matplotlib)
# # Example with Matplotlib
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

# node_features_np = node_features.cpu().numpy()
# node_features_2d = TSNE(n_components=2).fit_transform(node_features_np)

# plt.scatter(node_features_2d[:, 0], node_features_2d[:, 1])
# plt.title("t-SNE Visualization of Node Features")
# plt.show()