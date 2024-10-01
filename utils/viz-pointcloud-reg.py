import numpy as np
import open3d as o3d
import os
import pickle

def normalize_point_cloud(xyz):
    """Normalize the point cloud by shifting its center to the origin (0, 0, 0)."""
    centroid = np.mean(xyz, axis=0)
    return xyz - centroid, centroid

def transform_points(points, transform):
    """Apply a 4x4 transformation matrix to 3D points."""
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (transform @ homogeneous_points.T).T[:, :3]
    return transformed_points

def process_point_clouds(source_points, target_points, source_to_target_transform):
    """
    Normalize source points, transform target points to source frame,
    and adjust the source-to-target transform.
    
    :param source_points: numpy array of shape (N, 3)
    :param target_points: numpy array of shape (M, 3)
    :param source_to_target_transform: numpy array of shape (4, 4)
    :return: tuple (normalized_source, target_in_source_frame, adjusted_transform)
    """
    # Step 1: Normalize source points
    normalized_source, source_centroid = normalize_point_cloud(source_points)
    
    # Step 2: Create transformation matrix for source normalization
    source_norm_transform = np.eye(4)
    source_norm_transform[:3, 3] = -source_centroid
    
    # Step 3: Adjust source-to-target transform
    adjusted_transform = source_to_target_transform @ np.linalg.inv(source_norm_transform)
    
    # Step 4: Transform target points to normalized source frame
    target_in_source_frame = transform_points(target_points, np.linalg.inv(adjusted_transform))
    
    return normalized_source, target_in_source_frame, adjusted_transform


def visualize_point_clouds(source_points, target_points, transform):   #######Transform from source to target
    # Convert source points to target frame
    # transform = np.linalg.inv(transform)

    source_homogeneous = np.hstack((source_points, np.ones((source_points.shape[0], 1))))
    transformed_source = (transform @ source_homogeneous.T).T[:, :3]

    # Create Open3D point cloud objects
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(transformed_source)
    source_pcd.paint_uniform_color([1, 0, 0])  # Red color for source

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    target_pcd.paint_uniform_color([0, 0, 1])  # Blue color for target

    # Visualize point clouds
    o3d.visualization.draw_geometries([source_pcd, target_pcd],
                                      window_name="Aligned Point Clouds",
                                      width=800, height=600)

# Example usage
if __name__ == "__main__":
    base_dir = '/home/eavise3d/3DMatch_FCGF_Feature_32_transform'
    with open(os.path.join(base_dir, 'train_3dmatch', '2.pkl'), 'rb') as f:
        data = pickle.load(f)
    source_points = data.get('xyz_0')  # numpy array, for instance
    target_points = data.get('xyz_1')  # string or other type
    transform = data.get('gt_pose')

    # target_points = transform_target_to_source_frame(source_points, target_points)  

    # Process the point clouds
    normalized_source, target_in_source_frame, adjusted_transform = process_point_clouds(
        source_points, target_points, transform
    )
    centroid = np.mean(source_points, axis=0)
    source_points = source_points - centroid
    visualize_point_clouds(normalized_source, target_in_source_frame, adjusted_transform)