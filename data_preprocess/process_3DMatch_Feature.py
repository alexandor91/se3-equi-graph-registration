import os
import glob
import torch
import pickle
import numpy as np
import open3d as o3d
from tqdm import tqdm
import MinkowskiEngine as ME
from model import load_model
from util.trajectory import read_trajectory
from scipy.spatial import cKDTree

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def compute_correspondences_and_labels(xyz_0, xyz_1, feat_0, feat_1, gt_pose, sim_threshold=0.9, dist_threshold=0.06):
    """
    Compute correspondences and labels between source and target point clouds.
    
    Args:
        xyz_0 (np.ndarray): Source point cloud, shape [N, 3].
        xyz_1 (np.ndarray): Target point cloud, shape [M, 3].
        feat_0 (np.ndarray): Source feature descriptors, shape [N, D].
        feat_1 (np.ndarray): Target feature descriptors, shape [M, D].
        gt_pose (np.ndarray): Ground truth transformation matrix, shape [4, 4].
        sim_threshold (float): Threshold for dot product similarity.
        dist_threshold (float): Threshold for Euclidean distance in the transformed space.

    Returns:
        dict: Contains 'corr' and 'labels' arrays.
              'corr' - Nx2 array where each row is a (source_idx, target_idx) correspondence.
              'labels' - Binary labels array of shape [N], where 1 indicates a valid correspondence.
    """
    # Transform source points to the target frame
    xyz_0_h = np.concatenate([xyz_0, np.ones((xyz_0.shape[0], 1))], axis=1)  # Homogeneous coordinates
    xyz_0_transformed = (gt_pose @ xyz_0_h.T).T[:, :3]  # Transform and remove the homogeneous coordinate
    
    # Initialize correspondence and label arrays
    correspondences = []
    labels = np.zeros(xyz_0.shape[0], dtype=np.int32)
    
    # Compute the dot product similarity matrix between all source and target feature descriptors
    feat_similarity = feat_0 @ feat_1.T  # Shape [N, M]
    
    for src_idx in range(xyz_0.shape[0]):
        # Find potential matches in target based on similarity threshold
        similarity_mask = feat_similarity[src_idx] >= sim_threshold
        potential_matches = np.where(similarity_mask)[0]
        
        if len(potential_matches) == 0:
            # If no target points meet the similarity threshold, skip to next source point
            continue
        
        # Calculate distances from transformed source point to target points
        distances = np.linalg.norm(xyz_0_transformed[src_idx] - xyz_1[potential_matches], axis=1)
        
        # Filter potential matches based on distance threshold
        valid_matches = potential_matches[distances <= dist_threshold]
        
        if len(valid_matches) > 0:
            # If both similarity and distance conditions are met
            target_idx = valid_matches[np.argmin(distances[distances <= dist_threshold])]
            correspondences.append([src_idx, target_idx])
            labels[src_idx] = 1
        else:
            # If only the distance condition is met with relaxed threshold
            relaxed_distances = np.linalg.norm(xyz_0_transformed[src_idx] - xyz_1, axis=1)
            if np.min(relaxed_distances) <= dist_threshold * 1.5:  # Relaxed threshold
                target_idx = np.argmin(relaxed_distances)
                correspondences.append([src_idx, target_idx])
                labels[src_idx] = 0  # Relaxed match

    return {'corr': np.array(correspondences), 'labels': labels}

def compute_correspondences(feat_0, feat_1, xyz_0, xyz_1, gt_pose, feat_threshold=0.8, dist_threshold=0.05, relaxed_dist_threshold=0.1):
    # Apply GT pose to transform xyz_0 into target frame
    xyz_0_transformed = (gt_pose[:3, :3] @ xyz_0.T + gt_pose[:3, 3:4]).T
    
    # Initialize correspondence and label arrays
    corr = np.zeros((len(xyz_0), 2), dtype=int)
    labels = np.zeros(len(xyz_0), dtype=int)
    
    # Create a KDTree for the target point cloud for fast nearest-neighbor search
    tree = cKDTree(xyz_1)
    
    for i, (feat_src, point_src_trans) in enumerate(zip(feat_0, xyz_0_transformed)):
        # Find the nearest neighbor in target for each transformed source point
        dist, j = tree.query(point_src_trans)
        
        # Check distance threshold and feature similarity
        feat_similarity = np.dot(feat_src, feat_1[j])
        
        if dist < dist_threshold and feat_similarity > feat_threshold:
            # Both distance and feature similarity conditions are met
            corr[i] = [i, j]
            labels[i] = 1
        elif dist < relaxed_dist_threshold:
            # Only the distance condition (relaxed threshold) is met
            corr[i] = [i, j]
            labels[i] = 0

    return corr, labels

def do_single_pair_evaluation(feature_path, set_name, traj):
    result = {}
    trans_gth = np.linalg.inv(traj.pose)
    i = traj.metadata[0]
    j = traj.metadata[1]
    name_i = "%s_%03d" % (set_name, i)
    name_j = "%s_%03d" % (set_name, j)

    data_i = np.load(os.path.join(feature_path, name_i + ".npz"))
    coord_i, points_i, feat_i = data_i['xyz'], data_i['points'], data_i['feature']

    data_j = np.load(os.path.join(feature_path, name_j + ".npz"))
    coord_j, points_j, feat_j = data_j['xyz'], data_j['points'], data_j['feature']

    result['file_0'] = name_i
    result['file_1'] = name_j
    result['xyz_0'] = coord_i
    result['xyz_1'] = coord_j
    result['feat_0'] = feat_i
    result['feat_1'] = feat_j
    result['gt_pose'] = trans_gth

    # Compute correspondences and labels
    correspondences = compute_correspondences_and_labels(
        coord_i, coord_j, feat_i, feat_j, trans_gth)

    result.update(correspondences)
    return result

if __name__ == '__main__':
    OVERLAP_RATIO = 0.3
    data_root = '/media/HDD0/lzl/workspace/FCGF/data/threedmatch'
    train_files = ['/media/HDD0/lzl/workspace/FCGF/config/train_3dmatch.txt']

    out_folder = '/media/HDD0/lzl/workspace/FCGF/data/3DMatch_Feature_32'

    # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.
    num_feats = 1
    model_n_out = 32
    bn_momentum = 0.05
    normalize_feature = True
    conv1_kernel_size = 7
    model_checkpoint = '/media/HDD0/lzl/workspace/FCGF/checkpoints/3DMatch_32_0.05_2019-08-16_19-21-47.pth'

    # load the trained model
    checkpoint = torch.load(model_checkpoint)
    Model = load_model('ResUNetBN2C')
    model = Model(
        num_feats,
        model_n_out,
        bn_momentum=bn_momentum,
        normalize_feature=normalize_feature,
        conv1_kernel_size=conv1_kernel_size,
        D=3)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)

    for train_file in tqdm(train_files):
        subset_names = open(train_file).read().split()
        files = []
        vo_results = {}
        voxel_size = 0.05

        for name in subset_names:
            fname = name + "*%.2f.txt" % OVERLAP_RATIO
            fnames_txt = glob.glob(data_root + "/" + fname)
            assert len(
                fnames_txt) > 0, f"Make sure that the path {data_root} has data {fname}"
            for fname_txt in fnames_txt:
                with open(fname_txt) as f:
                    content = f.readlines()
                fnames = [x.strip().split() for x in content]
                for fname in fnames:
                    files.append([fname[0], fname[1]])

        ##############################################################
        # get the paris results for 3DMatch train dataset
        # Process each pair as before, but with correspondence and label calculations
        for idx in tqdm(range(len(files))):
            # Load data
            file0 = os.path.join(data_root, files[idx][0])
            file1 = os.path.join(data_root, files[idx][1])
            data0 = np.load(file0)
            data1 = np.load(file1)
            xyz0 = data0["pcd"]
            xyz1 = data1["pcd"]
            train_result = {}

            # Prepare features as in your original code
            feats_0, feats_1 = [], []
            feats_0.append(np.ones((len(xyz0), 1)))
            feats_0 = np.hstack(feats_0)
            feats_1.append(np.ones((len(xyz1), 1)))
            feats_1 = np.hstack(feats_1)

            # Voxelize xyz and get feats
            with torch.no_grad():
                coords_0 = np.floor(xyz0 / voxel_size)
                coords_0, inds_0 = ME.utils.sparse_quantize(
                    coords_0, return_index=True)
                coords_0 = ME.utils.batched_coordinates([coords_0])
                return_coords_0 = xyz0[inds_0]
                feats_0 = feats_0[inds_0]
                feats_0 = torch.tensor(feats_0, dtype=torch.float32)
                coords_0 = torch.tensor(coords_0, dtype=torch.int32)
                stensor_0 = ME.SparseTensor(
                    feats_0, coordinates=coords_0, device=device)
                feat0 = model(stensor_0).F.detach().cpu().numpy()

                coords_1 = np.floor(xyz1 / voxel_size)
                coords_1, inds_1 = ME.utils.sparse_quantize(
                    coords_1, return_index=True)
                coords_1 = ME.utils.batched_coordinates([coords_1])
                return_coords_1 = xyz1[inds_1]
                feats_1 = feats_1[inds_1]
                feats_1 = torch.tensor(feats_1, dtype=torch.float32)
                coords_1 = torch.tensor(coords_1, dtype=torch.int32)
                stensor_1 = ME.SparseTensor(
                    feats_1, coordinates=coords_1, device=device)
                feat1 = model(stensor_1).F.detach().cpu().numpy()

            traj = read_trajectory(os.path.join(
                source_path, set_name + "-train/gt.log"))
            assert len(traj) > 0, "Empty trajectory file"
            # Calculate correspondences and labels
            corr, labels = compute_correspondences(
                feat0, feat1, return_coords_0, return_coords_1, np.linalg.inv(traj[idx]) #########traj[idx] should be transform from source to target
            )

            # Save results
            train_result['file_0'] = os.path.basename(file0)
            train_result['file_1'] = os.path.basename(file1)
            train_result['xyz_0'] = return_coords_0
            train_result['xyz_1'] = return_coords_1
            train_result['feat_0'] = feat0
            train_result['feat_1'] = feat1
            train_result['corr'] = corr
            train_result['labels'] = labels
            train_result['gt_pose'] = traj[idx]

            # Save train result to pickle
            out_name = os.path.basename(train_file).split('.')[0]
            out_path = os.path.join(out_folder, out_name)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            pcd_path = os.path.join(out_path, f'{idx}.pkl')
            with open(pcd_path, 'wb') as f:
                pickle.dump(train_result, f, pickle.HIGHEST_PROTOCOL)


        ##############################################################
        # get the paris results for 3DMatch test dataset
        source_path = '/media/HDD0/lzl/workspace/FCGF/data/threedmatch_test'
        feature_path = '/media/HDD0/lzl/workspace/FCGF/features_tmp'

        with open(os.path.join(feature_path, "list.txt")) as f:
            sets = f.readlines()
            sets = [x.strip().split() for x in sets]
        assert len(sets) > 0, "Empty list file!"

        cnt = 0
        for s in tqdm(sets):
            set_name = s[0]
            traj = read_trajectory(os.path.join(
                source_path, set_name + "-evaluation/gt.log"))
            assert len(traj) > 0, "Empty trajectory file"
            for i in range(len(traj)):
                test_result = do_single_pair_evaluation(
                    feature_path, set_name, traj[i])

                # output
                out_name = 'test_3dmatch'
                out_path = os.path.join(out_folder, out_name)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                pcd_path = os.path.join(out_path, '%s.pkl' % cnt)
                with open(pcd_path, 'wb') as f:
                    pickle.dump(test_result, f, pickle.HIGHEST_PROTOCOL)

                cnt += 1