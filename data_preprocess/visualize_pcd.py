import os
import glob
import pickle
import numpy as np
import open3d as o3d
from tqdm import tqdm


def save_point_cloud(points, filename):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, point_cloud, write_ascii=True)


if __name__ == '__main__':
    OVERLAP_RATIO = 0.3
    data_root = '/media/HDD0/lzl/workspace/FCGF/data/threedmatch'
    configs_name = ['/media/HDD0/lzl/workspace/FCGF/config/val_3dmatch.txt']

    out_folder = '/media/HDD0/lzl/workspace/FCGF/data/visualization_ply'

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

            # save pcd for visualization
            out_name = os.path.basename(config_name).split('.')[0]
            out_path = os.path.join(out_folder, out_name)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            file0_name = os.path.basename(files[idx][0]).split('.')[0]
            file0_path = os.path.join(out_path, '%s.ply' % file0_name)
            save_point_cloud(xyz0, file0_path)

            file1_name = os.path.basename(files[idx][1]).split('.')[0]
            file1_path = os.path.join(out_path, '%s.ply' % file1_name)
            save_point_cloud(xyz1, file1_path)

            print('')
