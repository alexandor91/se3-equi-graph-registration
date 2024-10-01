import os
import random
from glob import glob

import torch

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import api


base_dir = '/home/eavise3d/Downloads/rgbd_dataset_freiburg1_xyz'

filename = os.path.join(base_dir, 'result.pth')  #####data load, read in
# data = {}
print('############')
data = torch.load(filename)
# print(data)
src_frame_fea = data['feats0_results']['keypoints']
src_frame_descriptor= data['feats0_results']['descriptors']
src_frame_points_scores = data['feats0_results']['keypoint_scores']
#################################################################
tar_frame_fea = data['feats1_results']['keypoints']
tar_frame_descriptor = data['feats1_results']['descriptors']
tar_frame_points_scores = data['feats1_results']['keypoint_scores']

print(data['feats0_results']['keypoints'].shape)
print(data['matches01']['matches1'].shape)
print(data['feats1_results']['keypoints'].shape)
print(src_frame_points_scores.shape)

# geom = torch.tensor(points.transpose())[None,:].double()
# feat = torch.randint(0, 20, (1, geom.shape[1],1)).double()

feats1 = torch.unsqueeze(src_frame_descriptor, 0).cuda() #torch.randn(2, 32, 32).cuda()
coors1 = torch.unsqueeze(src_frame_fea, 0).cuda() #torch.randn(2, 32, 3).cuda()
#rel_pos1 = torch.randint(0, 4, (2, 32, 32)).cuda()
# mask1  = src_frame_points_scores.bool().cuda()
mask1  = torch.ones(1, 2048).bool()

feats2 = tar_frame_descriptor.cuda() #torch.randn(2, 32, 32).cuda()
coors2 = tar_frame_fea.cuda() #torch.randn(2, 32, 3).cuda()
#rel_pos2 = torch.randint(0, 4, (2, 32, 32)).cuda()
mask2  = tar_frame_points_scores.bool().cuda()

# atom_feats = torch.randn(2, 32, 64)
# coors = torch.randn(2, 32, 3)
# mask  = torch.ones(2, 32).bool()
print(feats1.shape[0])
print(coors1.shape[0])
# print(data['matches01'])
print('$$$$$$$$$$$$$$$$$$$$')
print(data['matches01']['matches0'].shape)
print(data['matches01']['matches1'].shape)
print(data['matches01']['matching_scores0'].shape)
print(data['matches01']['matching_scores1'].shape)
print(data['matches01']['matches'].shape)

print(data['matches01']['scores'].shape)

pair_idxs = data['matches01']['matches'].cuda()
match_scores = data['matches01']['scores'].cuda()

matched_src_frame_fea = src_frame_fea[pair_idxs[:, 0]].cuda()
matched_src_frame_descriptor = src_frame_descriptor[pair_idxs[:, 0]].cuda()
matched_src_frame_points_scores = src_frame_points_scores[pair_idxs[:, 0]].cuda() #torch.zeros_like(pair_idxs[:, 0])

matched_tar_frame_fea = tar_frame_fea[pair_idxs[:, 1]].cuda() #torch.zeros_like(pair_idxs[:, 0])
matched_tar_frame_descriptor = tar_frame_descriptor[pair_idxs[:, 1]].cuda() #torch.zeros_like(pair_idxs[:, 0])
matched_tar_frame_points_scores = tar_frame_points_scores[pair_idxs[:, 1]].cuda() #torch.zeros_like(pair_idxs[:, 0])


print('&&&&&&&&&&')
print(matched_src_frame_descriptor.shape)
print(matched_src_frame_points_scores.shape)


# print(matched_src_frame_fea.shape)
# for i, item in enumerate(pair_idxs):
#     matched_src_frame_fea.append(src_frame_fea[item[0]])
#     matched_src_frame_descriptor.append(src_frame_descriptor[item[0]])
#     matched_src_frame_points_scores.append(src_frame_points_scores[item[0]])

#     matched_tar_frame_fea.append(tar_frame_fea[item[1]])
#     matched_tar_frame_descriptor.append(tar_frame_fea[item[1]])
#     matched_tar_frame_points_scores.append(tar_frame_points_scores[item[1]])

refined_coors1 = torch.zeros_like(coors1).cuda()
refined_coors2 = torch.zeros_like(coors1).cuda()

print(matched_src_frame_fea.shape)
print(matched_src_frame_descriptor.shape)
print(matched_src_frame_points_scores.shape)


class TUMDataset(Dataset):
    """
        Dataset class for the AutoFocus Task. Requires for each image, its depth ground-truth and
        segmentation mask
        Args:
            :- dataset_name -: str
            :- split -: split ['train', 'val', 'test']
    """
    def __init__(self, config, dataset_name, split=None):
        self.split = split
        self.config = config
        base_dir = ''

        path_images = os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name) #config['Dataset']['paths']['path_images']
        path_depths = os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, ) #config['Dataset']['paths']['path_depths']
        #path_segmentations = os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_segmentations'])
        print(path_images)
        print(path_depths)

        #self.paths_segmentations = get_total_paths(path_segmentations, config['Dataset']['extensions']['ext_segmentations'])

        assert (self.split in ['train', 'test', 'val']), "Invalid split!"
        assert (len(self.paths_images) == len(self.paths_depths)), "Different number of instances between the input and the depth maps"
        #assert (len(self.paths_images) == len(self.paths_segmentations)), "Different number of instances between the input and the segmentation maps"
        assert (config['Dataset']['splits']['split_train']+config['Dataset']['splits']['split_test']+config['Dataset']['splits']['split_val'] == 1), "Invalid splits (sum must be equal to 1)"
        # check for segmentation

        # utility func for splitting
        self.paths_images, self.paths_depths = get_splitted_dataset(config, self.split, dataset_name, self.paths_images, self.paths_depths)


    def __len__(self):
        """
            Function to get the number of images using the given list of images
        """
        return len(self.paths_images)

    def __getitem__(self, idx):
        """
            Getter function in order to get the triplet of images / depth maps and segmentation masks
        """
        print("iteration begins!")
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.transform_image(torch.load(self.paths_images[idx]))
        print("color image ######")

        #segmentation = self.transform_seg(Image.open(self.paths_segmentations[idx]))

        # exit(0)
        return src_pts, tar_pts, gt_rel_pose
