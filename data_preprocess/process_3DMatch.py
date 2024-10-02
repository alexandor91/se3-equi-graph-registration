import os
import glob
import pickle
import numpy as np
import open3d as o3d
from tqdm import tqdm



def random_ds_pc(point_cloud, num_points):
    if len(point_cloud) <= num_points:
        print('pcd num less than fixed num')
        return point_cloud
    
    indices = np.random.choice(len(point_cloud), num_points, replace=True)
    ds_pcd = point_cloud[indices]

    return ds_pcd


if __name__ == '__main__':
    OVERLAP_RATIO = 0.3
    data_root = '/media/HDD0/lzl/workspace/FCGF/data/threedmatch'
    configs_name = ['/media/HDD0/lzl/workspace/FCGF/config/train_3dmatch.txt',
                    '/media/HDD0/lzl/workspace/FCGF/config/val_3dmatch.txt']
    
    out_folder = '/media/HDD0/lzl/workspace/FCGF/data/3DMatch_VO'

    for config_name in configs_name:
        subset_names = open(config_name).read().split()
        files = []
        vo_results = {}
        voxel_size = 0.025
        pcd_num = 2048

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

        # get the paris pcd for 3DMatch dataset
        for idx in tqdm(range(len(files))):
            file0 = os.path.join(data_root, files[idx][0])
            file1 = os.path.join(data_root, files[idx][1])
            data0 = np.load(file0)
            data1 = np.load(file1)
            xyz0 = data0["pcd"]
            xyz1 = data1["pcd"]

            # Voxel downsample pcd
            pcd_0 = o3d.geometry.PointCloud()
            pcd_0.points = o3d.utility.Vector3dVector(xyz0)
            voxel_pcd_0 = pcd_0.voxel_down_sample(voxel_size)

            pcd_1 = o3d.geometry.PointCloud()
            pcd_1.points = o3d.utility.Vector3dVector(xyz1)
            voxel_pcd_1 = pcd_1.voxel_down_sample(voxel_size)

            # Random downsample pcd to fixed num pcd
            voxel_pcd_0 = random_ds_pc(np.asarray(voxel_pcd_0.points), pcd_num)
            voxel_pcd_1 = random_ds_pc(np.asarray(voxel_pcd_1.points), pcd_num)

            # get vo results
            vo_results['src_pcd'] = voxel_pcd_0
            vo_results['tar_pcd'] = voxel_pcd_1

            # Set output path for vo_results
            out_name = os.path.basename(config_name).split('.')[0]
            out_path = os.path.join(out_folder, out_name)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            pcd_path = os.path.join(out_path, '%s.pkl' % idx)
            with open(pcd_path, 'wb') as f:
                pickle.dump(vo_results, f, pickle.HIGHEST_PROTOCOL)
