import numpy as np
import os
import pickle
from scipy.spatial.transform import Rotation as R

def quaternion_to_matrix(q):
    """Convert quaternion to a 4x4 transformation matrix."""
    rot = R.from_quat(q[:4])
    matrix = np.eye(4)
    matrix[:3, :3] = rot.as_matrix()
    matrix[:3, 3] = q[4:]  # Translation
    return matrix

def calculate_pose_error(gt_pose, pred_pose):
    """Calculate the rotation and translation error between two 4x4 poses."""
    # Translation error (in meters converted to centimeters)
    translation_error = np.linalg.norm(gt_pose[:3, 3] - pred_pose[:3, 3]) * 100  # m to cm

    # Rotation error in radians, then converted to degrees
    rotation_diff = gt_pose[:3, :3].T @ pred_pose[:3, :3]  # Relative rotation
    rot_error = np.arccos(np.clip((np.trace(rotation_diff) - 1) / 2, -1.0, 1.0))  # acos for rotation angle
    rotation_error = np.degrees(rot_error)  # Convert to degrees

    return rotation_error, translation_error

def registration_recall(gt_pose, pred_pose, src_pts, tgt_pts, tau=0.09):
    """Calculate the registration recall based on the equation in the provided image."""
    # Apply the predicted transformation to the source points
    src_transformed = (pred_pose[:3, :3] @ src_pts.T).T + pred_pose[:3, 3]

    # Compute the Euclidean distance between transformed source and target points
    distances = np.linalg.norm(src_transformed - tgt_pts, axis=1)

    # Count True Positive Matches (below threshold tau)
    true_positives = np.sum(distances < tau)

    # Recall: sqrt(TP / Total ground truth points)
    recall = np.sqrt(true_positives / len(src_pts))

    # Precision: TP / Total predicted points (source transformed points)
    precision = true_positives / len(src_transformed) if len(src_transformed) > 0 else 0.0

    return recall, precision

def evaluate_pairwise_frames(gt_file_list, pred_file_list, gt_dir, pred_dir, save_dir):
    assert len(gt_file_list) == len(pred_file_list), "Ground truth and prediction file lists must have the same length."

    rotation_errors = []
    translation_errors = []
    recalls = []
    precisions = []
    f1_scores = []

    for gt_file, pred_file in zip(gt_file_list, pred_file_list):
        gt_file_path = os.path.join(gt_dir, gt_file)
        pred_file_path = os.path.join(pred_dir, pred_file)

        # Load the ground truth .pkl file containing gt_pose, src_pts, tar_pts
        with open(gt_file_path, 'rb') as f:
            gt_data = pickle.load(f)

        gt_pose = gt_data['gt_pose']  # 4x4 ground truth pose matrix
        src_pts = gt_data['xyz_0']  # Source point cloud
        tgt_pts = gt_data['xyz_1']  # Target point cloud

        # Load predicted pose (from corresponding .txt file in pred_dir)
        with open(pred_file_path, 'r') as pred_file:
            pred_data = list(map(float, pred_file.readline().strip().split()))
            pred_pose = quaternion_to_matrix(pred_data)

        # Calculate pose errors
        rot_err, trans_err = calculate_pose_error(gt_pose, pred_pose)
        rotation_errors.append(rot_err)
        translation_errors.append(trans_err)

        # Calculate registration recall
        recall = registration_recall(gt_pose, pred_pose, src_pts, tgt_pts)
        recalls.append(recall)

        # Calculate precision, F1 score (dummy example for now, modify based on your use case)
        precision = np.mean([recall])  # Placeholder for precision calculation
        precisions.append(precision)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
        f1_scores.append(f1_score)

    # Average metrics across all pairwise frames
    avg_rot_err = np.mean(rotation_errors)
    avg_trans_err = np.mean(translation_errors)
    avg_recall = np.mean(recalls)
    avg_precision = np.mean(precisions)
    avg_f1_score = np.mean(f1_scores)

    print(f"Average Rotation Error: {avg_rot_err:.4f} degrees")
    print(f"Average Translation Error: {avg_trans_err:.4f} cm")
    print(f"Average Registration Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1_score:.4f}")

    # Save the results (if needed)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "evaluation_results.txt"), "w") as result_file:
        result_file.write(f"Average Rotation Error: {avg_rot_err:.4f} degrees\n")
        result_file.write(f"Average Translation Error: {avg_trans_err:.4f} cm\n")
        result_file.write(f"Average Registration Recall: {avg_recall:.4f}\n")
        result_file.write(f"Average F1 Score: {avg_f1_score:.4f}\n")

# # Example usage:
# gt_file_list = ["0001.pkl", "0002.pkl", "0003.pkl"]  # List of ground truth files
# pred_file_list = ["0001.txt", "0002.txt", "0003.txt"]  # List of prediction files

# gt_dir = "./ground_truth"  # Directory with ground truth files
# pred_dir = "./predictions"  # Directory with predicted results
# save_dir = "./results"  # Directory to save evaluation results

# evaluate_pairwise_frames(gt_file_list, pred_file_list, gt_dir, pred_dir, save_dir)

