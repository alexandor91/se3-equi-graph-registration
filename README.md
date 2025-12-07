# Equi-GSPR: Equivariant SE(3) Graph Network Model for Sparse Point Cloud Registration

## Introduction

This repository contains the implementation of our Equi-GSPR: Equivariant SE(3) Graph Network Model for Sparse Point Cloud Registration. Our model is designed to process and align 3D point cloud data from various datasets, including 3DMatch and KITTI.
markdown

## System Overview

Below is an overview of our EGNN model architecture:

![Model Overview](assets/model-overview.png)

[Read the full paper here](https://eccv.ecva.net/virtual/2024/poster/944)


## Environment Setup
The code is tested on **pyg (Pytorch-Geometric)** **2.4.0**, **python 3.8**, **Pytorch 2.0.0**, **cuda11.8**, **GPU RAM** at least 8GB with batch size 1 on **GTX 2080** above.
Noted, my current code implementation with batch size one only consumes less than **0.9GB** RAM for 2048 points on GPU!!! and the batch size by default is set to one, other than one  may result in some training error, I will fix the batch size bug soon, please stay tuned, and batch size one can be enough for the current training as data size is not that big. The code can be ported onto edge device easily to support mobile applications.

To set up the environment for this project, we use Conda. Follow these steps:

1. Make sure you have Conda installed. If not, you can download it from [here](https://docs.conda.io/en/latest/miniconda.html).

2. Clone this repository, All required packages are specified in the `environment.yml` file. 

$conda env create -f environment.yml
$conda activate egnn-test

## Raw Data download

1. To run this project, you'll need to download the following raw data for the next step processing:

- [3DMatch](https://drive.google.com/file/d/1b8yA9AqJ0iBTfn9dhrK84JLG3CMGQRF3/view)
- [KITTI](https://drive.google.com/file/d/160htMU6rIOqGJIehqam9p5xBnWd-ri2g/view?usp=sharing)

2. After uncompression, you should have seen 'fragments', 'gt_results', under the root dir of the data folder,

3. Download 3DMatch raw fragments from the official website [3DMatch](https://3dmatch.cs.princeton.edu/), (PointDSC outdated), uncompress '7-Scenes.zip', then use `datasets/cal_overlap.py` from [D3Feat](https://github.com/XuyangBai/D3Feat?tab=readme-ov-file#refs) to process the downloaded data, which selects all the point cloud fragment pairs having more than 30% overlap, to filter out some low overlap pairs. and copy it into the same data folder level as 'fragments'. To run the processing script, 'fragments' +  'gt_results' + 'threedmatch' are mandatory.

4. Copy `3DMatch_Feature.py` from the `data_preprocess` folder into the data root dir, run `data_preprocess` folder to process. After running, all data pairs are saved in each pkl file

5. Use 'split_dataset_train_val' from 'datasets' folder to split train, val, and create test filelist txt files, put txt at the level of  pkl data folder (not inside folder with pkl!). change data [ath in training code.

## Data Processing for pair pkl files
 
For the training dataset processing, we provide scripts in the `data_preprocess` folder:

- `3DMatch_Feature.py`: For processing 3DMatch data, configure the directory based on the command in the function
- `process_kitti.py`: For processing KITTI dataset, configure the directory based on the command in the function

## Custom Data Processing
For self-processing data, please check the scripts in 'data_preprocess' folder for each individual training data processing:
If you own dataset is an ordered sequence of point cloud frames, just reuse the same KITTI processing script to process the sequentail point cloud frames, So the source and target scans are using $ i$th and $ i+1$th frame, respectively.
- `process_kitti.py`: For processing a sequential dataset.<br> 

For processing the KITTI dataset. Otherwise, if your point cloud frames are unordered, please refer to the 3D Match script to process, yet you have to establish the correspondence between source and target point scans, with a minimum 30% point overlapping between source and target scans, otherwise, we refer you to use public library like Open3D, PCL (Point Cloud Library), scikit-learn, through KDTree or Octree to create source and target frame correspondence with engouh point overlappings. Original 3DMatch already processed it for use. For further scan pair match, you can refer to the PointDSC repository to process the feature descriptors, [FPFH](https://github.com/XuyangBai/PointDSC/blob/master/misc/cal_fpfh.py), [FCGF](https://github.com/XuyangBai/PointDSC/blob/master/misc/cal_fcgf.py), as most of our data preprocessing codes are adapted based on their codes. 

- `3DMatch_Feature.py`: For processing paired scan dataset with enough overlapping > 30%. We use too conditions to determined the gt correspondence from source to target points, the transformed source point position distance to the target point is smaller than a threshold along with the point feature descriptors dot product similairty bigger than a threshold, label as one, otherwise, we only use point neighbouring search to find the match point pair, label as zero, 

## Training

To train the EGNN model, run the following script `3dmatch_train_egnn_with_batch.py` in the `src` folder for training of 3DMatch, batch size recommended no larger than 16:
$python src/3dmatch_train_egnn_with_batch.py

to train on kitti, run:
$python src/kitti_train_egnn_with_batch.py

One more thing to the training of custom dataset training, you can set "use_pointnet"  flag in the train model code to true, so that the model will train the model in end2end way from input point cloud scan pair to feature descriptor extraction, and until to equi-gnn regression, as custom dataset scenes may have some gap to indoor 3D Match or KITTI outdoor datasets. But indeed some more training time and tuning of layer hyper-params may be needed for this end2end training, and we also recommend you to use pre-trained  [PointTransformerV2](https://github.com/Pointcept/PointTransformerV2), [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3), the point transformer can be used as encoder for point feature descriptor extraction, to replace pointnet encoder in the code. By using the pre-trained -point transformer encoder weights fine-tuned on custom dataset you can mitigate the data gap, and it helps to converge fast based on our recent tests.

tensorboard logs are exported under "./runs" directory relative to the run script.

## Evaluation
Put the saved best checkpoint into the 'src/checkpoints' folder,

Use 'eval_egnn_metrics', set the "checkpointpath" variable in the main function to the specific checkpoint path you put, then run it to load the test data for evaluation.

$python src/eval_egnn_metrics.py

Average metric results data save in txt file and printed on the terminal where it is run.
The metric reuslts may ahve a bit fluctuation but around 1.4 degree for avg rotation error, and 4.5 cm for translation error. 

## Citation

If you find our work useful in your research, please kindly consider adding the following citation:

```bibtex
@inproceedings{kang2025equi,
  title={Equi-GSPR: Equivariant SE (3) Graph Network Model for Sparse Point Cloud Registration},
  author={Kang, Xueyang and Luan, Zhaoliang and Khoshelham, Kourosh and Wang, Bing},
  booktitle={European Conference on Computer Vision},
  pages={149--167},
  year={2025},
  organization={Springer}
}
