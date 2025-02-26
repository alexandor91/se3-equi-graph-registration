import os
import random
import numpy as np
import pickle

def load_pkl(file_path):
    """Load a .pkl file."""
    with open(file_path, 'rb') as f:
        data = np.load(f)
        # data = pickle.load(f)
    return data

def split_dataset(directory, train_ratio=0.90, seed=42):
    """Split the dataset into train and validation sets based on the specified ratio."""
    # Get all the .pkl files from the directory
    pkl_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')]
    ##################for test split ###########
    # pkl_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pkl')], \
    #                 key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    # # # Shuffle the list to ensure randomness
    random.seed(seed)
    random.shuffle(pkl_files)
    
    # Calculate the split index
    split_index = int(len(pkl_files) * train_ratio)
    
    # Split into train and validation sets
    train_files = pkl_files[:split_index]
    val_files = pkl_files[split_index:]
    
    return train_files, val_files

def save_file_list(file_list, output_file):
    """Save the file list into a .txt file."""
    with open(output_file, 'w') as f:
        for file in file_list:
            # Write only the file name (not the full path) if needed
            f.write(os.path.basename(file) + '\n')

def main():
    # Directory containing the .pkl files
    pkl_directory = '/media/eavise3d/新加卷/Datasets/eccv-data-0126/kitti/kitti/dataset'
    folder = 'fcgf_test'
    # Split ratio (train/val)
    train_ratio = 1.0
    
    # Output .txt files for the train and validation splits
    train_output = 'train_files.txt'
    val_output = 'val_files.txt'
    test_output = 'test_files.txt'
    
    # Split the dataset
    train_files, val_files = split_dataset(os.path.join(pkl_directory, folder), train_ratio=train_ratio)
    
    # Save the file names to .txt files
    # save_file_list(train_files, os.path.join(pkl_directory, train_output))
    # save_file_list(val_files, os.path.join(pkl_directory, val_output))

    save_file_list(train_files, os.path.join(pkl_directory, test_output))

    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print("File lists saved successfully.")

if __name__ == '__main__':
    main()
