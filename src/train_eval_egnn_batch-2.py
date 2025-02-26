import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionPoseRegression(nn.Module):
    def __init__(self, egnn: nn.Module, num_nodes: int = 2048, hidden_nf: int = 35, device='cuda:0'):
        super(CrossAttentionPoseRegression, self).__init__()
        self.egnn = egnn  # The shared EGNN network
        self.hidden_nf = hidden_nf  # Hidden feature dimension (35)
        self.num_nodes = num_nodes
        self.device = torch.device(device)  # Ensure device is a torch.device object

        # Move network to the specified device
        self.to(self.device)

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

    def _ensure_tensor_device(self, tensor, device=None):
        """
        Ensure tensor is on the specified device.
        If no device is specified, use the model's device.
        """
        device = device or self.device
        return tensor.to(device)

    def _preprocess_input(self, tensor, squeeze_batch=False):
        """
        Preprocess input tensor to ensure correct device and dimensionality.
        
        Args:
            tensor (torch.Tensor): Input tensor
            squeeze_batch (bool): Whether to squeeze batch dimension if it's 1
        
        Returns:
            torch.Tensor: Processed tensor
        """
        # Ensure tensor is on the correct device
        tensor = self._ensure_tensor_device(tensor)
        
        # Squeeze batch dimension if it's 1 and squeeze_batch is True
        if squeeze_batch and tensor.dim() == 3 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        
        return tensor

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
        - quaternion: [batch, 4] or [4]
        - translation: [batch, 3] or [3]
        - total_corr_loss: scalar loss
        """
        # Determine if input is batched
        is_batched = h_src.dim() == 3
        batch_size = h_src.size(0) if is_batched else 1

        # Preprocess inputs
        h_src = self._preprocess_input(h_src)
        x_src = self._preprocess_input(x_src)
        h_tgt = self._preprocess_input(h_tgt)
        x_tgt = self._preprocess_input(x_tgt)

        # Preprocess edges and attributes
        edges_src = [edge.to(self.device) if edge is not None else None for edge in (edges_src if isinstance(edges_src, list) else [edges_src])]
        edges_tgt = [edge.to(self.device) if edge is not None else None for edge in (edges_tgt if isinstance(edges_tgt, list) else [edges_tgt])]
        
        edge_attr_src = self._ensure_tensor_device(edge_attr_src) if edge_attr_src is not None else None
        edge_attr_tgt = self._ensure_tensor_device(edge_attr_tgt) if edge_attr_tgt is not None else None

        # Preprocess corr and labels
        corr = self._ensure_tensor_device(corr)
        labels = self._ensure_tensor_device(labels)

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
                    h_src_i, x_src_i, edges_src[0], edge_attr_src,
                    h_tgt_i, x_tgt_i, edges_tgt[0], edge_attr_tgt,
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
                h_src, x_src, edges_src[0], edge_attr_src,
                h_tgt, x_tgt, edges_tgt[0], edge_attr_tgt,
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
        corr_loss = F.mse_loss(corr_similarity, labels*torch.ones_like(corr_similarity))

        # Move MLP layers to the same device as input
        self.mlp_compress.to(h_src.device)
        self.mlp_pose.to(h_src.device)

        # Compress features to 128 dimensions
        compressed_h_src = self.mlp_compress(h_src.transpose(0, 1)).transpose(0, 1)
        compressed_h_tgt = self.mlp_compress(h_tgt.transpose(0, 1)).transpose(0, 1)

        compressed_h_src_norm = F.normalize(compressed_h_src, p=2, dim=-1)
        compressed_h_tgt_norm = F.normalize(compressed_h_tgt, p=2, dim=-1)

        # Compute similarity matrix (dot product)
        sim_matrix = torch.mm(compressed_h_src_norm, compressed_h_tgt_norm.t())  # Shape: [128, 128]

        # Compute rank loss
        u, s, v = torch.svd(sim_matrix)
        rank_loss = F.mse_loss(s[:128], torch.ones(128).to(sim_matrix.device))

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
        print(combined_features_flat.shape)
        # combined_features_flat = combined_features_flat.view(batch_size, -1, input_dim)
        pose = self.mlp_pose(combined_features_flat)  # Shape: [7]

        # Separate quaternion and translation
        quaternion = pose[:4]  # Shape: [4]
        translation = pose[4:]  # Shape: [3]

        return quaternion, translation, total_corr_loss

# Example usage
if __name__ == '__main__':
    # Dummy EGNN class for demonstration
    class DummyEGNN(nn.Module):
        def forward(self, h, x, edges, edge_attr):
            return h, x

    # Create a dummy EGNN instance
    dummy_egnn = DummyEGNN()

    # Create model
    model = CrossAttentionPoseRegression(dummy_egnn)

    # # Example inputs (single item)
    # h_src = torch.randn(2048, 35)
    # x_src = torch.randn(2048, 35)
    # edges_src = None
    # edge_attr_src = None

    # h_tgt = torch.randn(2048, 35)
    # x_tgt = torch.randn(2048, 35)
    # edges_tgt = None
    # edge_attr_tgt = None

    # corr = torch.randint(0, 2048, (100, 2))
    # labels = torch.rand(100)

    # # Test single item forward pass
    # quaternion, translation, loss = model(
    #     h_src, x_src, edges_src, edge_attr_src,
    #     h_tgt, x_tgt, edges_tgt, edge_attr_tgt,
    #     corr, labels
    # )

    # Example inputs (batched)
    h_src_batch = torch.randn(2, 2048, 35)
    x_src_batch = torch.randn(2, 2048, 35)
    edges_src_batch = None
    edge_attr_src_batch = None

    h_tgt_batch = torch.randn(2, 2048, 35)
    x_tgt_batch = torch.randn(2, 2048, 35)
    edges_tgt_batch = None
    edge_attr_tgt_batch = None

    corr_batch = torch.stack([
        torch.cat([torch.zeros(100, 1), torch.randint(0, 2048, (100, 2))], dim=1),
        torch.cat([torch.ones(100, 1), torch.randint(0, 2048, (100, 2))], dim=1)
    ])
    labels_batch = torch.rand(2, 100)

    # Test batched forward pass
    quaternions, translations, loss = model(
        h_src_batch, x_src_batch, edges_src_batch, edge_attr_src_batch,
        h_tgt_batch, x_tgt_batch, edges_tgt_batch, edge_attr_tgt_batch,
        corr_batch, labels_batch
    )

    print("Quaternions shape:", quaternions.shape)
    print("Translations shape:", translations.shape)
    print("Loss:", loss.item())