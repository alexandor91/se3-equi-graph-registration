import pickle
import os
import open3d as o3d
import numpy as np


def load_pkl_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    src_keypts = data['src_keypts'].numpy()
    tgt_keypts = data['tgt_keypts'].numpy()
    gt_pose = data['gt_pose'].numpy()

    # Remove batch dimension for source, target keypoints and ground truth pose
    src_keypts = np.squeeze(src_keypts, axis=0)  # (1, N, 3) -> (N, 3)
    tgt_keypts = np.squeeze(tgt_keypts, axis=0)  # (1, N, 3) -> (N, 3)
    gt_pose = np.squeeze(gt_pose, axis=0)  # (1, 4, 4) -> (4, 4)

    return src_keypts, tgt_keypts, gt_pose


def visualize_point_clouds(src_keypts, tgt_keypts):
    """
    Visualize the source and target point clouds.
    Args:
        src_keypts: Source keypoints (N, 3).
        tgt_keypts: Target keypoints (M, 3).
    """
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_keypts)
    src_pcd.paint_uniform_color([1, 0, 0])  # Red color for source points

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_keypts)
    tgt_pcd.paint_uniform_color([0, 1, 0])  # Green color for target points

    o3d.visualization.draw_geometries([src_pcd, tgt_pcd], window_name='Source and Target Point Clouds')


def apply_transform(pts, trans):
    """
    Apply a transformation matrix to 3D points.
    Args:
        pts: (N, 3) numpy array representing 3D points.
        trans: (4, 4) transformation matrix.
    Returns:
        Transformed 3D points as a (N, 3) numpy array.
    """
    R = trans[:3, :3]  # Extract rotation matrix (3x3)
    T = trans[:3, 3]  # Extract translation vector (3x1)
    pts = pts @ R.T + T  # Apply the transformation: pts * R^T + T
    return pts


def visualize_aligned_point_clouds(aligned_src_keypts, tgt_keypts):
    """
    Visualize the aligned source and target point clouds.
    Args:
        aligned_src_keypts: Aligned source keypoints (N, 3).
        tgt_keypts: Target keypoints (M, 3).
    """
    aligned_src_pcd = o3d.geometry.PointCloud()
    aligned_src_pcd.points = o3d.utility.Vector3dVector(aligned_src_keypts)
    aligned_src_pcd.paint_uniform_color([0, 0, 1])  # Blue color for aligned source points

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_keypts)
    tgt_pcd.paint_uniform_color([0, 1, 0])  # Green color for target points

    o3d.visualization.draw_geometries([aligned_src_pcd, tgt_pcd], window_name='Aligned Source and Target Point Clouds')


def save_point_clouds(src_keypts, tgt_keypts, aligned_src_keypts, save_dir, scene_name):
    """
    Save the source, target, and aligned source keypoints as .ply files in a specified directory.
    Args:
        src_keypts: Source keypoints (N, 3).
        tgt_keypts: Target keypoints (M, 3).
        aligned_src_keypts: Aligned source keypoints (N, 3) after transformation.
        save_dir: Directory where the .ply files will be saved.
        scene: The scene name used to label the file.
        index: The index used to label the file.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create Open3D PointCloud objects
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_keypts)
    o3d.io.write_point_cloud(os.path.join(save_dir, f'source_{scene_name}.ply'), src_pcd)

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_keypts)
    o3d.io.write_point_cloud(os.path.join(save_dir, f'target_{scene_name}.ply'), tgt_pcd)

    aligned_src_pcd = o3d.geometry.PointCloud()
    aligned_src_pcd.points = o3d.utility.Vector3dVector(aligned_src_keypts)
    o3d.io.write_point_cloud(os.path.join(save_dir, f'aligned_source_to_target{scene_name}.ply'), aligned_src_pcd)

    print(f"Saved point clouds to {save_dir}")


if __name__ == "__main__":
    # Set the path to the .pkl file
    pkl_path = '/home/zl/Desktop/PointDSC/data/kitti/dataset/processed_train_data/drive5-pair600_611.pkl'
    scene_name = pkl_path.split('/')[-1].split('.')[0]

    # Step 1: Load the saved .pkl file
    src_keypts, tgt_keypts, gt_pose = load_pkl_file(pkl_path)

    print(f"Source keypoints shape: {src_keypts.shape}")
    print(f"Target keypoints shape: {tgt_keypts.shape}")
    print(f"Ground truth transformation (gt_pose):\n{gt_pose}\n")

    # Step 2: Visualize the source and target point clouds
    print("Visualizing original source and target point clouds...")
    # visualize_point_clouds(src_keypts, tgt_keypts)

    # Step 3: Apply the ground truth transformation to source keypoints
    print("Applying ground truth transformation to source keypoints...")
    aligned_src_keypts = apply_transform(src_keypts, gt_pose)

    # Step 4: Visualize the aligned source and target point clouds
    print("Visualizing aligned source and target point clouds...")
    # visualize_aligned_point_clouds(aligned_src_keypts, tgt_keypts)

    # Step 5: Save the source, target, and aligned source point clouds as .ply files
    save_dir = '/home/zl/Desktop/PointDSC/outputs/KITTI/train_data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_point_clouds(src_keypts, tgt_keypts, aligned_src_keypts, save_dir, scene_name)
