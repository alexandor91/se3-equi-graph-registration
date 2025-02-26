import os
import glob
import pickle
from tqdm import tqdm
import numpy as np
import torch
import open3d as o3d
from SE3 import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def make_point_cloud(pts):
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd 

def extract_fpfh_features(pts, voxel_size):
    orig_pcd = make_point_cloud(pts)
    # voxel downsample 
    pcd = orig_pcd.voxel_down_sample(voxel_size=voxel_size)

    # estimate the normals and compute fpfh descriptor
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    fpfh_np = np.array(fpfh.data).T
    
    # save the data for training.
    # np.savez_compressed(
    #     os.path.join(save_path, pcd_path.replace('.npz', '_fpfh.npz')),
    #     points=np.array(orig_pcd.points).astype(np.float32),
    #     xyz=np.array(pcd.points).astype(np.float32),
    #     feature=fpfh_np.astype(np.float32),
    # )
    return np.array(pcd.points).astype(np.float32), fpfh_np.astype(np.float32)

if __name__ == '__main__':
    #Line 44 change root dir to the dir including kitti/dataset/fcgf/ 修改root路径为包含kitti/dataset/fcgf的路径
    #Line 45 change save path 修改存储路径，KITTI_FCGF_Features/KITTI_FPFH_Features
    #Line 51/167 change descriptor name fcgf or fpfh 修改为fcgf/fpfh
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = '/media/eavise3d/新加卷/Datasets/eccv-data-0126/kitti/kitti/dataset'
    out_folder = '/media/eavise3d/新加卷/Datasets/eccv-data-0126/kitti/kitti/dataset/fpfh_test'

    ############################ Process KITTI Training Dataset ############################
    # KKITTI training setting
    # split = 'train'
    # data_path = f"fcgf_test"
    # descriptor = 'fpfh'
    # use_mutual = False
    # num_node = 'all'
    # augment_axis = 3
    # augment_rotation = 1.0
    # augment_translation = 0.5
    # inlier_threshold = 0.10
    # in_dim = 6
    # downsample = 0.3
    # ids_list = []
    
    # for filename in os.listdir(f"{root}/{data_path}/"):
    #     ids_list.append(os.path.join(f"{root}/{data_path}/", filename))

    # for idx in tqdm(range(len(ids_list)), desc='processing the KITTI training data'):
    #     train_result ={}
    #     # load meta data
    #     file_name = ids_list[idx]
    #     data = np.load(file_name)
    #     src_keypts = data['xyz0']
    #     tgt_keypts = data['xyz1']
    #     src_features = data['features0']
    #     tgt_features = data['features1']
    #     if descriptor == 'fpfh':
    #         src_keypts, src_features = extract_fpfh_features(src_keypts, voxel_size=0.025)
    #         tgt_keypts, tgt_features = extract_fpfh_features(tgt_keypts, voxel_size=0.025)
    #         src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
    #         tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

    #     # compute ground truth transformation
    #     orig_trans = data['gt_trans']
    #     # data augmentation
    #     if split == 'train':
    #         src_keypts += np.random.rand(src_keypts.shape[0], 3) * 0.05
    #         tgt_keypts += np.random.rand(tgt_keypts.shape[0], 3) * 0.05
    #     aug_R = rotation_matrix(augment_axis, augment_rotation)
    #     aug_T = translation_matrix(augment_translation)
    #     aug_trans = integrate_trans(aug_R, aug_T)
    #     tgt_keypts = transform(tgt_keypts, aug_trans)
    #     gt_trans = concatenate(aug_trans, orig_trans)

    #     # select {num_node} numbers of keypoints
    #     N_src = src_features.shape[0]
    #     N_tgt = tgt_features.shape[0]
    #     src_sel_ind = np.arange(N_src)
    #     tgt_sel_ind = np.arange(N_tgt)
    #     # if num_node != 'all' and N_src > num_node:
    #     #     src_sel_ind = np.random.choice(N_src, num_node, replace=False)
    #     # if num_node != 'all' and N_tgt > num_node:
    #     #     tgt_sel_ind = np.random.choice(N_tgt, num_node, replace=False)

    #     if num_node == 'all':
    #         src_sel_ind = np.arange(N_src)
    #         tgt_sel_ind = np.arange(N_tgt)
    #     else:
    #         src_sel_ind = np.random.choice(N_src, num_node)
    #         tgt_sel_ind = np.random.choice(N_tgt, num_node)

    #     src_desc = src_features[src_sel_ind, :]
    #     tgt_desc = tgt_features[tgt_sel_ind, :]
    #     src_keypts = src_keypts[src_sel_ind, :]
    #     tgt_keypts = tgt_keypts[tgt_sel_ind, :]

    #     # construct the correspondence set by mutual nn in feature space.
    #     distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
    #     source_idx = np.argmin(distance, axis=1)
    #     if use_mutual:
    #         target_idx = np.argmin(distance, axis=0)
    #         mutual_nearest = (target_idx[source_idx] == np.arange(source_idx.shape[0]))
    #         corr = np.concatenate([np.where(mutual_nearest == 1)[0][:, None], source_idx[mutual_nearest][:, None]],
    #                               axis=-1)
    #     else:
    #         corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)

    #     # compute the ground truth label
    #     frag1 = src_keypts[corr[:, 0]]
    #     frag2 = tgt_keypts[corr[:, 1]]
    #     frag1_warp = transform(frag1, gt_trans)
    #     distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
    #     labels = (distance < inlier_threshold).astype(int)

    #     train_result['file_0'] = os.path.basename(file_name)
    #     train_result['file_1'] = os.path.basename(file_name)
    #     train_result['xyz_0'] = src_keypts.astype(np.float32)
    #     train_result['xyz_1'] = tgt_keypts.astype(np.float32)
    #     train_result['feat_0'] = src_desc
    #     train_result['feat_1'] = tgt_desc
    #     train_result['corr'] = corr
    #     train_result['labels'] = labels
    #     train_result['gt_pose'] = gt_trans


    #     # # save src points ###########################
    #     # out_path = "/home/xy/visual/input"
    #     # if not os.path.exists(out_path):
    #     #     os.makedirs(out_path)
    #     # np.savetxt(os.path.join(out_path, 'src.txt'), src_keypts, delimiter=',')
    #     # np.savetxt(os.path.join(out_path, 'tgt.txt'), tgt_keypts, delimiter=',')
        
    #     # src_transformed = transform(src_keypts, gt_trans)  
    #     # np.savetxt(os.path.join(out_path, 'src_transformed.txt'), src_transformed, delimiter=',')
    #     ##########################
    #     # output
    #     out_name = 'train_kitti'
    #     out_path = os.path.join(out_folder, out_name)
    #     if not os.path.exists(out_path):
    #         os.makedirs(out_path)

    #     pcd_path = os.path.join(out_path, '%s.pkl' % idx)
    #     with open(pcd_path, 'wb') as f:
    #         pickle.dump(train_result, f, pickle.HIGHEST_PROTOCOL)

    ############################# Process KITTI Testing Dataset ############################
    # KITTI testing setting
    split = 'test'
    data_path = f"fcgf_test"
    descriptor = 'fpfh'
    use_mutual = False
    num_node = 'all'
    augment_axis = 0
    augment_rotation = 0.0
    augment_translation = 0.0
    inlier_threshold = 0.60
    in_dim = 6
    downsample = 0.3
    ids_list = []

    for filename in os.listdir(f"{root}/{data_path}/"):
        ids_list.append(os.path.join(f"{root}/{data_path}/", filename))

    for idx in tqdm(range(len(ids_list)), desc='processing the KITTI testing data'):
        test_result = {}
        # load meta data
        file_name = ids_list[idx]
        data = np.load(file_name)
        src_keypts = data['xyz0']
        tgt_keypts = data['xyz1']
        src_features = data['features0']
        tgt_features = data['features1']
        if descriptor == 'fpfh':
            src_keypts, src_features = extract_fpfh_features(src_keypts, voxel_size=0.025)
            tgt_keypts, tgt_features = extract_fpfh_features(tgt_keypts, voxel_size=0.025)
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)
        
        # compute ground truth transformation
        orig_trans = data['gt_trans']
        # data augmentation
        if split == 'train':
            src_keypts += np.random.rand(src_keypts.shape[0], 3) * 0.05
            tgt_keypts += np.random.rand(tgt_keypts.shape[0], 3) * 0.05
        aug_R = rotation_matrix(augment_axis, augment_rotation)
        aug_T = translation_matrix(augment_translation)
        aug_trans = integrate_trans(aug_R, aug_T)
        tgt_keypts = transform(tgt_keypts, aug_trans)
        gt_trans = concatenate(aug_trans, orig_trans)

        # select {num_node} numbers of keypoints
        N_src = src_features.shape[0]
        N_tgt = tgt_features.shape[0]
        src_sel_ind = np.arange(N_src)
        tgt_sel_ind = np.arange(N_tgt)
        # if num_node != 'all' and N_src > num_node:
        #     src_sel_ind = np.random.choice(N_src, num_node, replace=False)
        # if num_node != 'all' and N_tgt > num_node:
        #     tgt_sel_ind = np.random.choice(N_tgt, num_node, replace=False)

        if num_node == 'all':
            src_sel_ind = np.arange(N_src)
            tgt_sel_ind = np.arange(N_tgt)
        else:
            src_sel_ind = np.random.choice(N_src, num_node)
            tgt_sel_ind = np.random.choice(N_tgt, num_node)

        src_desc = src_features[src_sel_ind, :]
        tgt_desc = tgt_features[tgt_sel_ind, :]
        src_keypts = src_keypts[src_sel_ind, :]
        tgt_keypts = tgt_keypts[tgt_sel_ind, :]

        # construct the correspondence set by mutual nn in feature space.
        distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
        source_idx = np.argmin(distance, axis=1)
        if use_mutual:
            target_idx = np.argmin(distance, axis=0)
            mutual_nearest = (target_idx[source_idx] == np.arange(source_idx.shape[0]))
            corr = np.concatenate(
                [np.where(mutual_nearest == 1)[0][:, None], source_idx[mutual_nearest][:, None]],
                axis=-1)
        else:
            corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)

        # compute the ground truth label
        frag1 = src_keypts[corr[:, 0]]
        frag2 = tgt_keypts[corr[:, 1]]
        frag1_warp = transform(frag1, gt_trans)
        distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        labels = (distance < inlier_threshold).astype(int)

        test_result['file_0'] = os.path.basename(file_name)
        test_result['file_1'] = os.path.basename(file_name)
        test_result['xyz_0'] = src_keypts.astype(np.float32)
        test_result['xyz_1'] = tgt_keypts.astype(np.float32)
        test_result['feat_0'] = src_desc
        test_result['feat_1'] = tgt_desc
        test_result['corr'] = corr
        test_result['labels'] = labels
        test_result['gt_pose'] = gt_trans

        # output
        out_name = 'test_kitti'
        out_path = os.path.join(out_folder, out_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        pcd_path = os.path.join(out_path, '%s.pkl' % idx)
        with open(pcd_path, 'wb') as f:
            pickle.dump(test_result, f, pickle.HIGHEST_PROTOCOL)
