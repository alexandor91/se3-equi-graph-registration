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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def do_single_pair_evaluation(feature_path, set_name, traj):
    result = {}
    trans_gth = np.linalg.inv(traj.pose)
    i = traj.metadata[0]
    j = traj.metadata[1]
    name_i = "%s_%03d" % (set_name, i)
    name_j = "%s_%03d" % (set_name, j)

    # coord and feat form a sparse tensor.
    data_i = np.load(os.path.join(feature_path, name_i + ".npz"))
    # points_i is the raw point cloud
    coord_i, points_i, feat_i = data_i['xyz'], data_i['points'], data_i['feature']

    data_j = np.load(os.path.join(feature_path, name_j + ".npz"))
    coord_j, points_j, feat_j = data_j['xyz'], data_j['points'], data_j['feature']

    result['file_0'] = name_i
    result['file_1'] = name_j
    result['xyz_0'] = coord_i
    result['xyz_1'] = coord_j
    result['feat_0'] = feat_i
    result['feat_1'] = feat_j
    result['pose_gt'] = trans_gth  # relative_pose: frame1->frame0

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
        for idx in tqdm(range(len(files))):
            file0 = os.path.join(data_root, files[idx][0])
            file1 = os.path.join(data_root, files[idx][1])
            data0 = np.load(file0)
            data1 = np.load(file1)
            xyz0 = data0["pcd"]
            xyz1 = data1["pcd"]
            train_result = {}

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

            train_result['file_0'] = os.path.basename(file0)
            train_result['file_1'] = os.path.basename(file1)
            train_result['xyz_0'] = return_coords_0
            train_result['xyz_1'] = return_coords_1
            train_result['feat_0'] = feat0
            train_result['feat_1'] = feat1

            # output
            out_name = os.path.basename(train_file).split('.')[0]
            out_path = os.path.join(out_folder, out_name)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            pcd_path = os.path.join(out_path, '%s.pkl' % idx)
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
