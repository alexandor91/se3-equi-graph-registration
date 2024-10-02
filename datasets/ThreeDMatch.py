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
                 augment_translation=0.01,
                 ):
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
        self.synthetic_pose_flag = True
        OVERLAP_RATIO = 0.3 
        DATA_FILES = {
            'train': '.train_3dmatch.txt',
            'val': '.val_3dmatch.txt',
        }
        # subset_names = open(DATA_FILES[split]).read().split()
        self.files = []
        self.length = 0
        self.normalize_use = False
        # for name in subset_names:
        #     fname = name + "*%.2f.txt" % OVERLAP_RATIO
        #     fnames_txt = glob.glob(root + f"/threedmatch/" + fname)
        #     assert len(fnames_txt) > 0, f"Make sure that the path {root} has data {fname}"
        #     for fname_txt in fnames_txt:
        #         with open(fname_txt) as f:
        #             content = f.readlines()
        #         fnames = [x.strip().split() for x in content]
        #         for fname in fnames:
        #             self.files.append([fname[0], fname[1]])
        #             self.length += 1
        self.gt_trans = {}
        self.src_filename = None
        self.tgt_filename = None
        # for scene in self.scene_list:
        #     scene_path = f'{self.root}/fragments/{scene}'
        #     gt_path = f'{self.root}/gt_result/{scene}-evaluation'
        #     for k, v in self.__loadlog__(gt_path).items():
        #         self.gt_trans[f'{scene}@{k}'] = v
        # Read all the filenames from the txt file
        if self.split == 'train':
            with open(os.path.join(self.root, 'train_files.txt'), 'r') as f:
                self.file_list = [line.strip() for line in f.readlines()]
        elif self.split == 'val':
            with open(os.path.join(self.root, 'val_files.txt'), 'r') as f:
                self.file_list = [line.strip() for line in f.readlines()]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'val'.")


    def __getitem__(self, index):
        # src_id, tgt_id = self.files[index][0], self.files[index][1]
        # if random.random() > 0.5:
        #     src_id, tgt_id = tgt_id, src_id
        file_name = self.file_list[index]

        # Load the .pkl file
        with open(os.path.join(self.root, 'train_3dmatch', file_name), 'rb') as f:
            data = pickle.load(f)

        # Extract the necessary parts of the data
        self.src_filename = data.get('file_0')
        self.tgt_filename = data.get('file_1')
        src_pts = data.get('xyz_0')  # numpy array, for instance
        tar_pts = data.get('xyz_1')  # string or other type
        gt_trans = data.get('gt_pose')
        # Convert the numpy array to a torch tensor, if needed
        # if isinstance(src_pts, np.ndarray) and isinstance(tar_pts, np.ndarray):
        #     src_pts = torch.tensor(src_pts, dtype=torch.float32)
        #     tar_pts = torch.tensor(tar_pts, dtype=torch.float32)
        self.length = self.length + 1
        # if self.length == 2:
        # time.sleep(5)  # Pause execution for 2 seconds

        src_features = data.get('feat_0')
        tgt_features = data.get('feat_1')
        
        # Normalizing for fpfh
        if self.descriptor == 'fpfh':
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        # orig_trans = self.gt_trans[key]
        corr = data.get('corr')
        labels = data.get('labels')

        N_src = src_features.shape[0]
        N_tgt = tgt_features.shape[0]

        sensor_origin = np.array([0, 0, 0])
        # Order the points by the ray length to the sensor origin for training convenience
        xyz0 = data["xyz_0"]
        feat0 = data["feat_0"]
        ray_lengths0 = np.linalg.norm(xyz0 - sensor_origin, axis=1)
        sorted_indices0 = np.argsort(ray_lengths0, kind = 'stable')
        data["xyz_0"] = xyz0[sorted_indices0]
        data["feat_0"] = feat0[sorted_indices0]
        # print("$$$$$$$$$ ordered src points $$$$$$$$")
        # Order the points by the ray length to the sensor origin for training convenience
        xyz1 = data["xyz_1"]
        feat1 = data["feat_1"]
        ray_lengths1 = np.linalg.norm(xyz1 - sensor_origin, axis=1)
        sorted_indices1 = np.argsort(ray_lengths1, kind = 'stable')
        data["xyz_1"] = xyz1[sorted_indices1]
        data["feat_1"] = feat1[sorted_indices1]
        # print("$$$$$$$$$ ordered tar points $$$$$$$$")
        # print(np.linalg.norm(data["xyz_1"] - sensor_origin, axis=1))

        if N_src < N_tgt:
            # Modify the "Corr" and "Labels" accordingly
            data["corr"] = data["corr"][sorted_indices0]        
            data["labels"] = data["labels"][sorted_indices0]
        elif N_tgt < N_src:
            # Modify the "Corr" and "Labels" accordingly
            data["corr"] = data["corr"][sorted_indices1]        
            data["labels"] = data["labels"][sorted_indices1]

        if self.synthetic_pose_flag == True:
            orig_trans = np.eye(4).astype(np.float32)
            src_pts += np.random.rand(src_pts.shape[0], 3) * 0.005
            tar_pts += np.random.rand(tar_pts.shape[0], 3) * 0.005
            aug_R = rotation_matrix(self.augment_axis, self.augment_rotation)
            aug_T = translation_matrix(self.augment_translation)
            aug_trans = integrate_trans(aug_R, aug_T)
            src_pts = transform(src_pts, aug_trans)            
            gt_trans = concatenate(aug_trans, orig_trans)

        if int(N_src) < int(self.num_node) or int(N_tgt) < int(self.num_node):
            # index += 1  # Skip this index and move to the next one
            if index >= len(self.file_list):
                raise IndexError("End of dataset reached after skipping.")
            # return None
        # Initialize sel_ind as an empty array (in case no condition is met)
        sel_ind = np.array([])

        if N_src <= N_tgt:
            sel_ind = np.random.choice(N_src, self.num_node)  # Randomly select self.num_node points from source
        else:
            sel_ind = np.random.choice(N_tgt, self.num_node)  # Randomly select self.num_node points from target

        ########### src target , crorrespondence iterate loop   ####################3
        corr = corr[sel_ind, :]            
        labels = labels[sel_ind]
        src_desc = src_features[corr[:, 0], :]
        tgt_desc = tgt_features[corr[:, 1], :]
        input_src_keypts = src_pts[corr[:, 0], :]
        input_tgt_keypts = tar_pts[corr[:, 1], :]

        if self.normalize_use:
            input_tgt_keypts = transform_target_to_source_frame(input_src_keypts, input_tgt_keypts)
            centroid = np.mean(input_src_keypts, axis=0)
            input_src_keypts = input_src_keypts - centroid            
        # distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
        # source_idx = np.argmin(distance, axis=1)
        # if self.use_mutual:
        #     target_idx = np.argmin(distance, axis=0)
        #     mutual_nearest = (target_idx[source_idx] == np.arange(source_idx.shape[0]))
        #     corr = np.concatenate([np.where(mutual_nearest == 1)[0][:,None], source_idx[mutual_nearest][:,None]], axis=-1)
        # else:
        #     corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)
        # if len(corr) < 10:
        #     return self.__getitem__(int(np.random.choice(self.__len__(),1)))
        
        # frag1 = src_keypts[corr[:, 0]]
        # frag2 = tgt_keypts[corr[:, 1]]
        # frag1_warp = transform(frag1, gt_trans)
        # distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        # labels = (distance < self.inlier_threshold).astype(np.int)

        # if self.in_dim == 6:
        #     corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts], axis=-1)
        #     corr_pos = corr_pos - corr_pos.mean(0)

        return corr.astype(np.float32), \
               labels.astype(np.float32), \
               input_src_keypts.astype(np.float32), \
               input_tgt_keypts.astype(np.float32), \
               src_desc.astype(np.float32), \
               tgt_desc.astype(np.float32), \
               gt_trans.astype(np.float32) 
               
    def __len__(self):
        return len(self.file_list)


class ThreeDMatchTest(data.Dataset):
    def __init__(self, 
                 root, 
                 descriptor='fcgf',
                 in_dim=6,
                 inlier_threshold=0.10,
                 num_node=2048, 
                 use_mutual=True,
                 downsample=0.03, 
                 augment_axis=0, 
                 augment_rotation=1.0,
                 augment_translation=0.01,
                 select_scene=None,
                 ):
        self.root = root
        self.descriptor = descriptor
        assert descriptor in ['fcgf', 'fpfh']
        self.in_dim = in_dim
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.downsample = downsample
        self.augment_axis = augment_axis
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation
        self.length = 0
        
        self.scene_list = [
            '7-scenes-redkitchen',
            'sun3d-home_at-home_at_scan1_2013_jan_1',
            'sun3d-home_md-home_md_scan9_2012_sep_30',
            'sun3d-hotel_uc-scan3',
            'sun3d-hotel_umd-maryland_hotel1',
            'sun3d-hotel_umd-maryland_hotel3',
            'sun3d-mit_76_studyroom-76-1studyroom2',
            'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
        ]
        if select_scene in self.scene_list:
            self.scene_list = [select_scene]
        
        self.gt_trans = {}
        self.src_filename = None
        self.tgt_filename = None
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
        # key = list(self.gt_trans.keys())[index]      
        # scene = key.split('@')[0]
        # Get the filename for this index
        file_name = self.test_file_list[index]

        # Load the .pkl file
        with open(os.path.join(self.root, 'test_3dmatch', file_name), 'rb') as f:
            data = pickle.load(f)
        # Extract the necessary parts of the data
        self.src_filename = data.get('file_0')
        self.tgt_filename = data.get('file_1')
        src_pts = data.get('xyz_0')  # numpy array, for instance
        tar_pts = data.get('xyz_1')  # string or other type
        gt_trans = data.get('pose_gt')   ## camera to world pose
        # Check if the number of points in src_pts or tar_pts is less than 2048
        if src_pts.shape[0] < self.num_node or tar_pts.shape[0] < self.num_node:
            print(f"Skipping index {index}: insufficient number of points (src_pts: {src_pts.shape[0]}, tar_pts: {tar_pts.shape[0]})")
            return None  #
        # Convert the numpy array to a torch tensor, if needed
        # if isinstance(src_pts, np.ndarray) and isinstance(tar_pts, np.ndarray):
        #     src_pts = torch.tensor(src_pts, dtype=torch.float32)
        #     tar_pts = torch.tensor(tar_pts, dtype=torch.float32
        src_features = data.get('feat_0')
        tgt_features = data.get('feat_1')

        # Normalizing for fpfh
        if self.descriptor == 'fpfh':
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        # orig_trans = self.gt_trans[key]
        corr = data.get('corr')
        labels = data.get('labels')


        N_src = src_features.shape[0]
        N_tgt = tgt_features.shape[0]

        sensor_origin = np.array([0, 0, 0])
        # Order the points by the ray length to the sensor origin for training convenience
        xyz0 = data["xyz_0"]
        feat0 = data["feat_0"]
        ray_lengths0 = np.linalg.norm(xyz0 - sensor_origin, axis=1)
        sorted_indices0 = np.argsort(ray_lengths0, kind = 'stable')
        data["xyz_0"] = xyz0[sorted_indices0]
        data["feat_0"] = feat0[sorted_indices0]
        # print("$$$$$$$$$ ordered src points $$$$$$$$")
        # print(np.linalg.norm(data["xyz_0"] - sensor_origin, axis=1))
        # Order the points by the ray length to the sensor origin for training convenience
        xyz1 = data["xyz_1"]
        feat1 = data["feat_1"]
        ray_lengths1 = np.linalg.norm(xyz1 - sensor_origin, axis=1)
        sorted_indices1 = np.argsort(ray_lengths1, kind = 'stable')
        data["xyz_1"] = xyz1[sorted_indices1]
        data["feat_1"] = feat1[sorted_indices1]
        # print("$$$$$$$$$ ordered tar points $$$$$$$$")
        # print(np.linalg.norm(data["xyz_1"] - sensor_origin, axis=1))

        if N_src < N_tgt:
            # Modify the "Corr" and "Labels" accordingly
            data["corr"] = data["corr"][sorted_indices0]        
            data["labels"] = data["labels"][sorted_indices0]
        elif N_tgt < N_src:
            # Modify the "Corr" and "Labels" accordingly
            data["corr"] = data["corr"][sorted_indices1]        
            data["labels"] = data["labels"][sorted_indices1]

        if int(N_src) < int(self.num_node) or int(N_tgt) < int(self.num_node):
            index += 1  # Skip this index and move to the next one
            if index >= len(self.test_file_list):
                raise IndexError("End of dataset reached after skipping.")

        # Initialize sel_ind as an empty array (in case no condition is met)
        sel_ind = np.array([])

        if N_src <= N_tgt:
            sel_ind = np.random.choice(N_src, self.num_node)  # Randomly select self.num_node points from source
        else:
            sel_ind = np.random.choice(N_tgt, self.num_node)  # Randomly select self.num_node points from target

        ########### src target , crorrespondence iterate loop   ####################3
        corr = corr[sel_ind, :]            
        labels = labels[sel_ind]
        src_desc = src_features[corr[:, 0], :]
        tgt_desc = tgt_features[corr[:,1], :]
        input_src_keypts = src_pts[corr[:, 0]]
        input_tgt_keypts = tar_pts[corr[:, 1]]


        distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
        source_idx = np.argmin(distance, axis=1)

        #######normalize to source frame as origin, to get rid of nan issues############
        if self.normalize_use:
            input_tgt_keypts = transform_target_to_source_frame(input_src_keypts, input_tgt_keypts)
            centroid = np.mean(input_src_keypts, axis=0)
            input_src_keypts = input_src_keypts - centroid         
        # if self.use_mutual:
        #     target_idx = np.argmin(distance, axis=0)
        #     mutual_nearest = (target_idx[source_idx] == np.arange(source_idx.shape[0]))
        #     corr = np.concatenate([np.where(mutual_nearest == 1)[0][:,None], source_idx[mutual_nearest][:,None]], axis=-1)
        # else:
        #     corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)
        
        # frag1 = src_keypts[corr[:, 0]]
        # frag2 = tgt_keypts[corr[:, 1]]
        # frag1_warp = transform(frag1, orig_trans)
        # distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        # labels = (distance < self.inlier_threshold).astype(np.int)

        # if self.in_dim == 6:
        #     corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts], axis=-1)
        #     corr_pos = corr_pos - corr_pos.mean(0)

        return corr.astype(np.float32), \
               labels.astype(np.float32), \
               input_src_keypts.astype(np.float32), \
               input_tgt_keypts.astype(np.float32), \
               src_desc.astype(np.float32), \
               tgt_desc.astype(np.float32), \
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

