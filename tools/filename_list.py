import os
import random


# # Directory containing the .pkl files
directory = "/media/eavise3d/新加卷/Datasets/eccv-data-0126/3DMatch/3DMatch_fcgf_feature_test/test_3dmatch"

# Output text file
output_txt_file = "test_files.txt"

# Get all .pkl filenames in the directory
pkl_filenames = [f for f in os.listdir(directory) if f.endswith(".pkl")]

# Write filenames to the text file
with open(output_txt_file, "w") as f:
    for filename in pkl_filenames:
        f.write(filename + "\n")

print(f"Saved {len(pkl_filenames)} filenames to {output_txt_file}")

# Output text files
# train_txt_file = "train_files.txt"
# val_txt_file = "val_files.txt"

# # Number of validation samples
# num_val_samples = 1000

# # Get all .pkl filenames in the directory
# pkl_filenames = [f for f in os.listdir(directory) if f.endswith(".pkl")]

# # Shuffle filenames randomly
# random.shuffle(pkl_filenames)

# # Split into train and validation
# val_filenames = pkl_filenames[:num_val_samples]  # First 1000 for validation
# train_filenames = pkl_filenames[num_val_samples:]  # The rest for training

# # Write validation filenames to val.txt
# with open(val_txt_file, "w") as f:
#     for filename in val_filenames:
#         f.write(filename + "\n")

# # Write training filenames to train.txt
# with open(train_txt_file, "w") as f:
#     for filename in train_filenames:
#         f.write(filename + "\n")

# print(f"Total files: {len(pkl_filenames)}")
# print(f"Saved {len(train_filenames)} train filenames to {train_txt_file}")
# print(f"Saved {len(val_filenames)} validation filenames to {val_txt_file}")