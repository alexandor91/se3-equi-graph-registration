# Copyright (c) Alexander.Kang alexander.kang@tum.de
# Equi-GSPR: Equivariant Graph Model for Sparse Point Cloud Registration
# Please cite the following papers if you use any part of the code.
import os
from os.path import join, exists
import pickle
import glob
import random
import torch.utils.data as data
# from utils.pointcloud import make_point_cloud, estimate_normal
#from utils.SE3 import *
import numpy as np
import random
import torch
import time

def rotation_matrix(num_axis, augment_rotation):
    """
    Sample rotation matrix along [num_axis] axis and [0 - augment_rotation] angle
    Input
        - num_axis:          rotate along how many axis
        - augment_rotation:  rotate by how many angle
    Output
        - R: [3, 3] rotation matrix
    """
    assert num_axis == 1 or num_axis == 3 or num_axis == 0
    if  num_axis == 0:
        return np.eye(3)
    angles = np.random.rand(3) * 2 * np.pi * augment_rotation
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    # R = Rx @ Ry @ Rz
    if num_axis == 1:
        return random.choice([Rx, Ry, Rz]) 
    return Rx @ Ry @ Rz

def translation_matrix(augment_translation):
    """
    Sample translation matrix along 3 axis and [augment_translation] meter
    Input
        - augment_translation:  translate by how many meters
    Output
        - t: [3, 1] translation matrix
    """
    T = np.random.rand(3) * augment_translation
    return T.reshape(3, 1)
    
def transform(pts, trans):
    """
    Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
    Input
        - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
    """
    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0,2,1) + trans[:, :3, 3:4]
        return trans_pts.permute(0,2,1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T

def decompose_trans(trans):
    """
    Decompose SE3 transformations into R and t, support torch.Tensor and np.ndarry.
    Input
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    """
    if len(trans.shape) == 3:
        return trans[:, :3, :3], trans[:, :3, 3:4]
    else:
        return trans[:3, :3], trans[:3, 3:4]
    
def integrate_trans(R, t):
    """
    Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
    Input
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    Output
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    """
    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1).to(R.device)
        else:
            trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4).to(R.device)
        else:
            trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans

def concatenate(trans1, trans2):
    """
    Concatenate two SE3 transformations, support torch.Tensor and np.ndarry.
    Input
        - trans1: [4, 4] or [bs, 4, 4], SE3 transformation matrix
        - trans2: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output:
        - trans1 @ trans2
    """    
    R1, t1 = decompose_trans(trans1)
    R2, t2 = decompose_trans(trans2)
    R_cat = R1 @ R2
    t_cat = R1 @ t2 + t1
    trans_cat = integrate_trans(R_cat, t_cat)
    return trans_cat

def normalize_point_cloud(xyz):
    """Normalize the point cloud by shifting its center to the origin (0, 0, 0)."""
    centroid = np.mean(xyz, axis=0)
    return xyz - centroid, centroid

def transform_target_to_source_frame(xyz_source, xyz_target):
    """
    Transform the target point cloud to be with respect to the source frame origin.
    
    :param xyz_source: numpy array of shape (N, 3) representing the source point cloud
    :param xyz_target: numpy array of shape (M, 3) representing the target point cloud
    :return: transformed target point cloud of shape (M, 3)
    """
    # Normalize source point cloud
    xyz_source_normalized, source_centroid = normalize_point_cloud(xyz_source)
    
    # Calculate the transformation from world to source frame
    # This is just a translation in this case
    world_to_source_transform = np.eye(4)
    world_to_source_transform[:3, 3] = -source_centroid
    
    # Calculate the inverse transformation (source to world)
    source_to_world_transform = np.linalg.inv(world_to_source_transform)
    
    # Apply the inverse transformation to the target point cloud
    xyz_target_homogeneous = np.hstack((xyz_target, np.ones((xyz_target.shape[0], 1))))
    xyz_target_transformed = (world_to_source_transform @ xyz_target_homogeneous.T).T[:, :3]
    
    return xyz_target_transformed


class ThreeDMatchTrainVal(data.Dataset):
    def __init__(self, 
                 root, 
                 split, 
                 descriptor='fcgf',
                 in_dim=6,
                 inlier_threshold=0.10,
                 num_node=2048, 
                 use_mutual=True,
                 downsample=0.03, 
                 augment_axis=1, 
                 augment_rotation=1.0,
                 augment_translation=0.01):
        self.root = root
        self.split = split
        self.descriptor = descriptor
        assert descriptor in ['fpfh', 'fcgf']
        self.in_dim = in_dim
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.downsample = downsample
        self.augment_axis = augment_axis
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation
        self.synthetic_pose_flag = False
        self.normalize_use = False

        # Load the file list based on the split
        if self.split == 'train':
            with open(os.path.join(self.root, 'train_files.txt'), 'r') as f:
                self.file_list = [line.strip() for line in f.readlines()]
        elif self.split == 'val':
            with open(os.path.join(self.root, 'val_files.txt'), 'r') as f:
                self.file_list = [line.strip() for line in f.readlines()]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'val'.")

    # def __getitem__(self, index):
    #     file_name = self.file_list[index]

    #     # Load data from .pkl file
    #     with open(os.path.join(self.root, 'train_3dmatch', file_name), 'rb') as f:
    #         data = pickle.load(f)

    #     # Extract data
    #     src_pts = data.get('xyz_0')  # Nx3
    #     tar_pts = data.get('xyz_1')  # Mx3
    #     src_features = data.get('feat_0')  # Nx32
    #     tgt_features = data.get('feat_1')  # Mx32
    #     corr = data.get('corr')  # Correspondence (Nx2), maps source index -> target index
    #     labels = data.get('labels')  # Binary labels (Nx1)
    #     gt_trans = data.get('gt_pose')  # 4x4 ground truth transformation

    #     # Normalize features if using FPFH descriptor
    #     if self.descriptor == 'fpfh':
    #         src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
    #         tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

    #     N_src = len(src_pts)
    #     N_tgt = len(tar_pts)

    #     # Sort points by ray length to the sensor origin
    #     sensor_origin = np.array([0, 0, 0])
    #     ray_lengths_src = np.linalg.norm(src_pts - sensor_origin, axis=1)
    #     sorted_indices_src = np.argsort(ray_lengths_src)
        
    #     ray_lengths_tgt = np.linalg.norm(tar_pts - sensor_origin, axis=1)
    #     sorted_indices_tgt = np.argsort(ray_lengths_tgt)

    #     # Reorder points and features based on ray lengths
    #     src_pts = src_pts[sorted_indices_src]
    #     src_features = src_features[sorted_indices_src]

    #     tar_pts = tar_pts[sorted_indices_tgt]
    #     tgt_features = tgt_features[sorted_indices_tgt]

    #     # Reorder `corr` based on the new source and target orders.
    #     # Remap source indices:
    #     corr[:, 0] = np.searchsorted(sorted_indices_src, corr[:, 0])  
    #     # Remap target indices:
    #     corr[:, 1] = np.searchsorted(sorted_indices_tgt, corr[:, 1])  

    #     # Reorder `labels` based on sorted source points:
    #     labels = labels[sorted_indices_src]

    #     # Ensure `self.num_node` doesn't exceed the number of available points
    #     sample_size = min(self.num_node, N_src, N_tgt)
    #     if sample_size < self.num_node:
    #         sample_size = self.num_node
    #         print("!!!Not enough sample points for the fixed number, so sampling with repetitions!!!")

    #     # Sample fixed points and adjust `corr` accordingly
    #     sel_src = np.random.choice(N_src, sample_size, replace=True)
    #     sel_tgt = np.random.choice(N_tgt, sample_size, replace=True)

    #     # Remap `corr` to fit the new sampled indices.
    #     # We need to map the original `corr` indices to the new sampled index range.
    #     corr[:, 0] = np.searchsorted(sel_src, corr[:, 0])  # Remap source ids
    #     corr[:, 1] = np.searchsorted(sel_tgt, corr[:, 1])  # Remap target ids

    #     # Sampled descriptors and points based on the new `corr`
    #     src_desc = src_features[sel_src]  # Sampled source descriptors
    #     tgt_desc = tgt_features[sel_tgt]  # Sampled target descriptors
    #     input_src_keypts = src_pts[sel_src]  # Sampled source keypoints
    #     input_tgt_keypts = tar_pts[sel_tgt]  # Sampled target keypoints

    #     # Data augmentation (if required)
    #     if self.synthetic_pose_flag:
    #         src_pts += np.random.rand(src_pts.shape[0], 3) * 0.005
    #         tar_pts += np.random.rand(tar_pts.shape[0], 3) * 0.005
    #         aug_R = rotation_matrix(self.augment_axis, self.augment_rotation)
    #         aug_T = translation_matrix(self.augment_translation)
    #         aug_trans = integrate_trans(aug_R, aug_T)
    #         src_pts = transform(src_pts, aug_trans)
    #         gt_trans = concatenate(aug_trans, np.eye(4).astype(np.float32))

    #     # Optional normalization
    #     if self.normalize_use:
    #         input_tgt_keypts = transform_target_to_source_frame(input_src_keypts, input_tgt_keypts)
    #         centroid = np.mean(input_src_keypts, axis=0)
    #         input_src_keypts -= centroid

    #     return corr.astype(np.float32), \
    #         labels.astype(np.float32), \
    #         input_src_keypts.astype(np.float32), \
    #         input_tgt_keypts.astype(np.float32), \
    #         src_desc.astype(np.float32), \
    #         tgt_desc.astype(np.float32), \
    #         gt_trans.astype(np.float32)

    def __getitem__(self, index):
        file_name = self.file_list[index]

        # Load data from .pkl file
        with open(os.path.join(self.root, 'train_3dmatch', file_name), 'rb') as f:
            data = pickle.load(f)

        # Extract data
        src_pts = data.get('xyz_0')  # Nx3
        tar_pts = data.get('xyz_1')  # Mx3
        src_features = data.get('feat_0')  # Nx32
        tgt_features = data.get('feat_1')  # Mx32
        corr = data.get('corr')  # Correspondence (Nx2)
        labels = data.get('labels')  # Binary labels (N,)
        gt_trans = data.get('gt_pose')  # 4x4 ground truth transformation

        # Normalize features if using FPFH descriptor
        if self.descriptor == 'fpfh':
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        N_src = len(src_pts)
        N_tgt = len(tar_pts)
        # Count the number of ones
        # num_ones = np.count_nonzero(labels == 1)

        # print(f"Number of ones: {num_ones}")
        
        # Sort the points by ray length to sensor origin for ordering
        sensor_origin = np.array([0, 0, 0])
        ray_lengths_src = np.linalg.norm(src_pts - sensor_origin, axis=1)
        sorted_indices_src = np.argsort(ray_lengths_src)
        
        ray_lengths_tgt = np.linalg.norm(tar_pts - sensor_origin, axis=1)
        sorted_indices_tgt = np.argsort(ray_lengths_tgt)

        # Create inverse mappings for sorted indices
        inverse_map_src = {old: new for new, old in enumerate(sorted_indices_src)}
        inverse_map_tgt = {old: new for new, old in enumerate(sorted_indices_tgt)}

        # Sample fixed number of points
        sample_size = self.num_node
        # if sample_size > N_src or sample_size > N_tgt:
        #     print("Warning: Not enough sample points for the fixed number, sampling with repetitions.")
        
        # Separate indices for positive and negative labels
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]

        # Sample 60% from positive labels and 40% from negative labels
        num_pos = int(self.num_node * 0.6)
        num_neg = self.num_node - num_pos

        # if len(pos_indices) < num_pos or len(neg_indices) < num_neg:
        #     print("Not enough positive or negative points to satisfy the 0.60-0.40 ratio. so repeating sampling will be used!")

        if len(pos_indices) < 5:
            sampled_indices = np.random.choice(len(labels), self.num_node, replace=True)
        else:
            pos_sampled = np.random.choice(pos_indices, num_pos, replace=True)
            neg_sampled = np.random.choice(neg_indices, num_neg, replace=True)
            # Combine positive and negative indices
            sampled_indices = np.concatenate([pos_sampled, neg_sampled])
        # Sample source points and features
        sampled_src_pts = src_pts[sampled_indices]
        sampled_src_features = src_features[sampled_indices]

        # Use the second column of corr to get corresponding target indices
        sampled_corr = corr[sampled_indices]  # Nx2
        sampled_tgt_indices = sampled_corr[:, 1].astype(int)  # Get target indices
        sampled_tgt_pts = tar_pts[sampled_tgt_indices]  # Get corresponding target points
        sampled_tgt_features = tgt_features[sampled_tgt_indices]  # Get target descriptors

        # Create remapping for source indices in the new range
        remap_src = {old_idx: new_idx for new_idx, old_idx in enumerate(sampled_indices)}

        # Create remapping for target indices based on unique sampled target indices
        unique_tgt_indices = np.unique(sampled_tgt_indices)
        remap_tgt = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_tgt_indices)}

        # Remap corr source indices to be in the range of self.num_node
        remapped_corr = np.zeros_like(sampled_corr)
        remapped_corr[:, 0] = [remap_src[src_idx] for src_idx in sampled_corr[:, 0]]  # Remap source
        remapped_corr[:, 1] = [remap_tgt[tgt_idx] for tgt_idx in sampled_corr[:, 1]]  # Remap target

        # print(remapped_corr)
        # Retrieve the labels for the resampled source points
        sampled_labels = labels[sampled_indices]

        # # Now remap target indices
        # remap_tgt = {old: new for new, old in enumerate(np.unique(orig_tgt_indices))}
        # sampled_corr[:, 1] = np.array([remap_tgt.get(idx, -1) for idx in sampled_corr[:, 1]])  # Safely map target indices
        
        # Data augmentation
        if self.synthetic_pose_flag:
            sampled_src_pts += np.random.rand(sample_size, 3) * 0.005
            sampled_tgt_pts += np.random.rand(sample_size, 3) * 0.005
            aug_R = rotation_matrix(self.augment_axis, self.augment_rotation)
            aug_T = translation_matrix(self.augment_translation)
            aug_trans = integrate_trans(aug_R, aug_T)
            sampled_src_pts = transform(sampled_src_pts, aug_trans)
            gt_trans = concatenate(aug_trans, np.eye(4).astype(np.float32))

        # Optional normalization
        if self.normalize_use:
            sampled_tgt_pts = transform_target_to_source_frame(sampled_src_pts, sampled_tgt_pts)
            centroid = np.mean(sampled_src_pts, axis=0)
            sampled_src_pts -= centroid

        return remapped_corr.astype(np.float32), \
            sampled_labels.astype(np.float32), \
            sampled_src_pts.astype(np.float32), \
            sampled_tgt_pts.astype(np.float32), \
            sampled_src_features.astype(np.float32), \
            sampled_tgt_features.astype(np.float32), \
            gt_trans.astype(np.float32)
               
    def __len__(self):
        return len(self.file_list)


class ThreeDMatchTest(data.Dataset):
    def __init__(self, 
                 root, 
                 split, 
                 descriptor='fcgf',
                 in_dim=6,
                 inlier_threshold=0.10,
                 num_node=2048, 
                 use_mutual=True,
                 downsample=0.03, 
                 augment_axis=1, 
                 augment_rotation=1.0,
                 augment_translation=0.01):
        self.root = root
        self.split = split
        self.descriptor = descriptor
        assert descriptor in ['fpfh', 'fcgf']
        self.in_dim = in_dim
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.downsample = downsample
        self.augment_axis = augment_axis
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation
        self.synthetic_pose_flag = False
        self.normalize_use = False
        # for scene in self.scene_list:
        #     scene_path = f'{self.root}/fragments/{scene}'
        #     gt_path = f'{self.root}/gt_result/{scene}-evaluation'
        #     for k, v in self.__loadlog__(gt_path).items():
        #         self.gt_trans[f'{scene}@{k}'] = v
        # Read all the filenames from the txt file
        with open(os.path.join(self.root, 'test_files.txt'), 'r') as f:
            self.test_file_list = [line.strip() for line in f.readlines()]

    def __getitem__(self, index):
        file_name = self.file_list[index]

        # Load data from .pkl file
        with open(os.path.join(self.root, 'train_3dmatch', file_name), 'rb') as f:
            data = pickle.load(f)

        # Extract data
        src_pts = data.get('xyz_0')  # Nx3
        tar_pts = data.get('xyz_1')  # Mx3
        src_features = data.get('feat_0')  # Nx32
        tgt_features = data.get('feat_1')  # Mx32
        corr = data.get('corr')  # Correspondence (Nx2)
        labels = data.get('labels')  # Binary labels (N,)
        gt_trans = data.get('gt_pose')  # 4x4 ground truth transformation

        # Normalize features if using FPFH descriptor
        if self.descriptor == 'fpfh':
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        N_src = len(src_pts)
        N_tgt = len(tar_pts)
        # Count the number of ones
        # num_ones = np.count_nonzero(labels == 1)

        # print(f"Number of ones: {num_ones}")
        
        # Sort the points by ray length to sensor origin for ordering
        sensor_origin = np.array([0, 0, 0])
        ray_lengths_src = np.linalg.norm(src_pts - sensor_origin, axis=1)
        sorted_indices_src = np.argsort(ray_lengths_src)
        
        ray_lengths_tgt = np.linalg.norm(tar_pts - sensor_origin, axis=1)
        sorted_indices_tgt = np.argsort(ray_lengths_tgt)

        # Create inverse mappings for sorted indices
        inverse_map_src = {old: new for new, old in enumerate(sorted_indices_src)}
        inverse_map_tgt = {old: new for new, old in enumerate(sorted_indices_tgt)}

        # Sample fixed number of points
        sample_size = self.num_node
        # if sample_size > N_src or sample_size > N_tgt:
        #     print("Warning: Not enough sample points for the fixed number, sampling with repetitions.")
        
        # Separate indices for positive and negative labels
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]

        # Sample 60% from positive labels and 40% from negative labels
        num_pos = int(self.num_node * 0.6)
        num_neg = self.num_node - num_pos

        # if len(pos_indices) < num_pos or len(neg_indices) < num_neg:
        #     print("Not enough positive or negative points to satisfy the 0.60-0.40 ratio. so repeating sampling will be used!")


        if len(pos_indices) < 5:
            sampled_indices = np.random.choice(len(labels), self.num_node, replace=True)
        else:
            pos_sampled = np.random.choice(pos_indices, num_pos, replace=True)
            neg_sampled = np.random.choice(neg_indices, num_neg, replace=True)
            # Combine positive and negative indices
            sampled_indices = np.concatenate([pos_sampled, neg_sampled])

        # Sample source points and features
        sampled_src_pts = src_pts[sampled_indices]
        sampled_src_features = src_features[sampled_indices]

        # Use the second column of corr to get corresponding target indices
        sampled_corr = corr[sampled_indices]  # Nx2
        sampled_tgt_indices = sampled_corr[:, 1].astype(int)  # Get target indices
        sampled_tgt_pts = tar_pts[sampled_tgt_indices]  # Get corresponding target points
        sampled_tgt_features = tgt_features[sampled_tgt_indices]  # Get target descriptors

        # Create remapping for source indices in the new range
        remap_src = {old_idx: new_idx for new_idx, old_idx in enumerate(sampled_indices)}

        # Create remapping for target indices based on unique sampled target indices
        unique_tgt_indices = np.unique(sampled_tgt_indices)
        remap_tgt = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_tgt_indices)}

        # Remap corr source indices to be in the range of self.num_node
        remapped_corr = np.zeros_like(sampled_corr)
        remapped_corr[:, 0] = [remap_src[src_idx] for src_idx in sampled_corr[:, 0]]  # Remap source
        remapped_corr[:, 1] = [remap_tgt[tgt_idx] for tgt_idx in sampled_corr[:, 1]]  # Remap target

        # print(remapped_corr)
        # Retrieve the labels for the resampled source points
        sampled_labels = labels[sampled_indices]

        # # Now remap target indices
        # remap_tgt = {old: new for new, old in enumerate(np.unique(orig_tgt_indices))}
        # sampled_corr[:, 1] = np.array([remap_tgt.get(idx, -1) for idx in sampled_corr[:, 1]])  # Safely map target indices
        
        # Data augmentation
        if self.synthetic_pose_flag:
            sampled_src_pts += np.random.rand(sample_size, 3) * 0.005
            sampled_tgt_pts += np.random.rand(sample_size, 3) * 0.005
            aug_R = rotation_matrix(self.augment_axis, self.augment_rotation)
            aug_T = translation_matrix(self.augment_translation)
            aug_trans = integrate_trans(aug_R, aug_T)
            sampled_src_pts = transform(sampled_src_pts, aug_trans)
            gt_trans = concatenate(aug_trans, np.eye(4).astype(np.float32))

        # Optional normalization
        if self.normalize_use:
            sampled_tgt_pts = transform_target_to_source_frame(sampled_src_pts, sampled_tgt_pts)
            centroid = np.mean(sampled_src_pts, axis=0)
            sampled_src_pts -= centroid

        return remapped_corr.astype(np.float32), \
            sampled_labels.astype(np.float32), \
            sampled_src_pts.astype(np.float32), \
            sampled_tgt_pts.astype(np.float32), \
            sampled_src_features.astype(np.float32), \
            sampled_tgt_features.astype(np.float32), \
            gt_trans.astype(np.float32)

                  

    def __len__(self):
        return len(self.test_file_list)
    
    def __loadlog__(self, gtpath):
        traj = {}
        with open(gtpath) as f:
            content = f.readlines()
        for i in range(len(content) // 5):
            idx = content[i * 5].strip().split()
            T = np.fromstring(' '.join([x.strip() for x in content[i * 5 + 1:i * 5 + 5]]), dtype=float, sep=' ').reshape(4, 4)
            traj[f"{idx[0]}_{idx[1]}"] = T
        return traj

if __name__ == "__main__":
    base_dir = '/home/eavise3d/3DMatch_FCGF_Feature_32_transform'
    # pkl_file = '5.pkl'
    mode = "train"
    if mode == "train":
        dset = ThreeDMatchTrainVal(root=base_dir, 
                            split='train',   
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
        
        print(len(dset))  
        for i in range(dset.__len__()):
            ret_dict = dset[i]
    if mode == "test":
        dset = ThreeDMatchTest(root=base_dir, 
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
        
        print(len(dset))  
        for i in range(dset.__len__()):
            ret_dict = dset[i]
