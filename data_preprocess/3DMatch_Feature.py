import os
import glob
import pickle
from tqdm import tqdm
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
# from utils.pointcloud import make_point_cloud, estimate_normal
from utils.SE3 import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def __loadlog__(gtpath):
    with open(os.path.join(gtpath, 'gt.log')) as f:
        content = f.readlines()
    result = {}
    i = 0
    while i < len(content):
        line = content[i].replace("\n", "").split("\t")[0:3]
        trans = np.zeros([4, 4])
        trans[0] = np.fromstring(content[i+1], dtype=float, sep=' \t')
        trans[1] = np.fromstring(content[i+2], dtype=float, sep=' \t')
        trans[2] = np.fromstring(content[i+3], dtype=float, sep=' \t')
        trans[3] = np.fromstring(content[i+4], dtype=float, sep=' \t')
        i = i + 5
        result[f'{int(line[0])}_{int(line[1])}'] = trans
    return result


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Line 44 change root dir to the dir including kitti/dataset/fcgf/ 修改root路径为包含kitti/dataset/fcgf的路径
    #Line 45 change save path out_folder 修改存储路径，KITTI_FCGF_Features/KITTI_FPFH_Features
    #Line 51/167 change descriptor name fcgf or fpfh 修改为fcgf/fpfh
    root = '/media/eavise3d/新加卷/Datasets/eccv-data-0126/3DMatch'
    out_folder = '/media/eavise3d/新加卷/Datasets/eccv-data-0126/3DMatch/3DMatch_fcgf_feature_test'
    make_training_data = False # True generate training set/ False generate test set
    # ############################ Process 3DMatch Training Dataset ############################
    if make_training_data: 
        # # output
        out_name = 'train_3dmatch'
        out_path = os.path.join(out_folder, out_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        split = 'train'
        # use point cloud pairs with more than 30% overlapping as the training set
        OVERLAP_RATIO = 0.3
        DATA_FILES = {'train': '/media/eavise3d/新加卷/Datasets/eccv-data-0126/3DMatch/misc/split/test_3dmatch.txt'}
        subset_names = open(DATA_FILES[split]).read().split()
        files = []
        length = 0

        # 3DMatch training setting
        descriptor = 'fcgf'
        use_mutual = False
        num_node = 'all'
        augment_axis = 3
        augment_rotation = 1.0
        augment_translation = 0.5
        inlier_threshold = 0.10
        in_dim = 6
        downsample = 0.03

        for name in subset_names:
            fname = name + "*%.2f.txt" % OVERLAP_RATIO
            fnames_txt = glob.glob(root + f"/threedmatch/" + fname)
            assert len(
                fnames_txt) > 0, f"Make sure that the path {root} has data {fname}"
            for fname_txt in fnames_txt:
                with open(fname_txt) as f:
                    content = f.readlines()
                fnames = [x.strip().split() for x in content]
                for fname in fnames:
                    files.append([fname[0], fname[1]])

        for idx in tqdm(range(len(files)), desc='processing the 3DMatch trainging data'):
            train_result = {}
            # load meta data
            src_id, tgt_id = files[idx][0], files[idx][1]
            if random.random() > 0.5:
                src_id, tgt_id = tgt_id, src_id
            # load point coordinates and pre-computed per-point local descriptors
            if descriptor == 'fcgf':
                src_data = np.load(
                    f"{root}/threedmatch_feat/{src_id}".replace('.npz', '_fcgf.npz'))
                tgt_data = np.load(
                    f"{root}/threedmatch_feat/{tgt_id}".replace('.npz', '_fcgf.npz'))
                src_keypts = src_data['xyz']
                tgt_keypts = tgt_data['xyz']
                src_features = src_data['feature']
                tgt_features = tgt_data['feature']

            elif descriptor == 'fpfh':
                src_data = np.load(
                    f"{root}/threedmatch_feat_fpfh/{src_id}".replace('.npz', '_fpfh.npz'))
                tgt_data = np.load(
                    f"{root}/threedmatch_feat_fpfh/{tgt_id}".replace('.npz', '_fpfh.npz'))
                src_keypts = src_data['xyz']
                tgt_keypts = tgt_data['xyz']
                src_features = src_data['feature']
                tgt_features = tgt_data['feature']
                np.nan_to_num(src_features)
                np.nan_to_num(tgt_features)
                src_features = src_features / \
                    (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
                tgt_features = tgt_features / \
                    (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)
            
            # compute ground truth transformation
            orig_trans = np.eye(4).astype(np.float32)
            # data augmentation (add data augmentation to original transformation)
            src_keypts += np.random.rand(src_keypts.shape[0], 3) * 0.005
            tgt_keypts += np.random.rand(tgt_keypts.shape[0], 3) * 0.005

            aug_R = rotation_matrix(augment_axis, augment_rotation)
            aug_T = translation_matrix(augment_translation)
            aug_trans = integrate_trans(aug_R, aug_T)
            tgt_keypts = transform(tgt_keypts, aug_trans)
            gt_pose = concatenate(aug_trans, orig_trans)

            # save src points ###########################
            # out_path = "/home/xy/visual/step1"
            # if not os.path.exists(out_path):
            #     os.makedirs(out_path)
            # np.savetxt(os.path.join(out_path, 'src.txt'), src_keypts, delimiter=',')
            # np.savetxt(os.path.join(out_path, 'tgt.txt'), tgt_keypts, delimiter=',')
            
            #######################################
            #######################################
            # src_transformed = transform(src_keypts, gt_pose)  
            # np.savetxt(os.path.join(out_path, 'src_transformed.txt'), src_transformed, delimiter=',')
            # np.savetxt(os.path.join(out_path, 'tgt_transformed.txt'), tgt_keypts, delimiter=',')
            #######################################
            

            # select {num_node} numbers of keypoints
            N_src = src_features.shape[0]
            N_tgt = tgt_features.shape[0]
            if num_node == 'all':
                src_sel_ind = np.arange(N_src)
                tgt_sel_ind = np.arange(N_tgt)
            else:
                src_sel_ind = np.random.choice(N_src, num_node)
                tgt_sel_ind = np.random.choice(N_tgt, num_node)
            src_desc = src_features[src_sel_ind, :]
            tgt_desc = tgt_features[tgt_sel_ind, :]
            src_keypts = src_keypts[src_sel_ind, :]
            tgt_keypts = tgt_keypts[tgt_sel_ind, :]
            if(N_src < 2048 or N_tgt < 2048):
                print(num_node, src_desc.shape, tgt_desc.shape, src_keypts.shape, tgt_keypts.shape)
            # print(num_node, src_desc.shape, tgt_desc.shape, src_keypts.shape, tgt_keypts.shape)
            
            
            # construct the correspondence set by nearest neighbor searching in feature space. single direction
            distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
            source_idx = np.argmin(distance, axis=1)
            source_dis = np.min(distance, axis=1)
            if use_mutual:
                target_idx = np.argmin(distance, axis=0)
                mutual_nearest = (target_idx[source_idx] == np.arange(source_idx.shape[0]))
                corr = np.concatenate([np.where(mutual_nearest == 1)[0][:, None], source_idx[mutual_nearest][:, None]], axis=-1)
            else:
                corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)


            # inlier_threshold = 0.5

            # similarity_matrix = src_desc @ tgt_desc.T  # Compute NxN similarity matrix

            # # Find nearest neighbors using feature similarity
            # source_idx = np.argmax(similarity_matrix, axis=1)  # Best match for each source feature

            # if use_mutual:
            #     target_idx = np.argmax(similarity_matrix, axis=0)  # Best match for each target feature
            #     mutual_nearest = (target_idx[source_idx] == np.arange(source_idx.shape[0]))  # Mutual check
                
            #     # Assign mutual correspondences; others remain as one-way correspondences
            #     corr_mutual = np.column_stack((np.where(mutual_nearest)[0], source_idx[mutual_nearest]))  # Mutual pairs
                
            #     # Ensure corr remains the same shape as src_pts
            #     full_corr = np.column_stack((np.arange(source_idx.shape[0]), source_idx))  # Default all pairs
            #     full_corr[mutual_nearest] = corr_mutual  # Replace mutual pairs

            #     corr = full_corr  # Assign back to corr
            # else:
            #     corr = np.column_stack((np.arange(source_idx.shape[0]), source_idx))  # One-way correspondence

            # # Transform source points using ground-truth extrinsic matrix
            # corr_tar = tgt_keypts[corr[:, 1]]  # Get corresponding target points
            # src_pt_warp = transform(src_keypts, gt_pose)  # Warp source points

            # # Compute error between transformed source points and their matched target points
            # distance = np.linalg.norm(src_pt_warp[corr[:, 0]] - corr_tar, axis=1)  # Compute per-point error
            # labels = (distance < inlier_threshold).astype(int)  # Label inliers (1) and

            # compute the ground truth label
            frag1 = src_keypts[corr[:, 0]]
            frag2 = tgt_keypts[corr[:, 1]]
            frag1_warp = transform(frag1, gt_pose)
            distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
            labels = (distance < inlier_threshold).astype(np.int32)
            
            train_result['file_0'] = os.path.basename(src_id)
            train_result['file_1'] = os.path.basename(tgt_id)
            train_result['xyz_0'] = src_keypts.astype(np.float32)
            train_result['xyz_1'] = tgt_keypts.astype(np.float32)
            train_result['feat_0'] = src_desc
            train_result['feat_1'] = tgt_desc
            train_result['corr'] = corr
            train_result['labels'] = labels
            train_result['gt_pose'] = gt_pose
            #
            # # save src points ###########################
            # out_path = "/home/xy/visual"
            # if not os.path.exists(out_path):
            #     os.makedirs(out_path)
            # np.savetxt(os.path.join(out_path, 'src.txt'), src_keypts, delimiter=',')
            # np.savetxt(os.path.join(out_path, 'tgt.txt'), tgt_keypts, delimiter=',')
            # print(gt_pose)
            # src_transformed = transform(src_keypts, gt_pose)  
            # np.savetxt(os.path.join(out_path, 'src_transformed.txt'), src_transformed, delimiter=',')
            # pcd_path = '/home/xy/visual/%s.pkl' % idx
            # print(pcd_path)
            # with open(pcd_path, 'wb') as f:
            #     pickle.dump(train_result, f, pickle.HIGHEST_PROTOCOL)
            ###########################
            
            pcd_path = os.path.join(out_path, '%s.pkl' % idx)
            with open(pcd_path, 'wb') as f:
                pickle.dump(train_result, f, pickle.HIGHEST_PROTOCOL)
    else:    
        # ############################ Process the 3DMatch Testing Dataset ############################
        # output
        out_name = 'test_3dmatch'
        out_path = os.path.join(out_folder, out_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        scene_list = [
            '7-scenes-redkitchen',
            'sun3d-home_at-home_at_scan1_2013_jan_1',
            'sun3d-home_md-home_md_scan9_2012_sep_30',
            'sun3d-hotel_uc-scan3',
            'sun3d-hotel_umd-maryland_hotel1',
            'sun3d-hotel_umd-maryland_hotel3',
            'sun3d-mit_76_studyroom-76-1studyroom2',
            'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
        ]

        # 3DMatch training setting
        descriptor = 'fpfh'
        augment_axis = 0
        augment_rotation = 0.0
        augment_translation = 0.0
        use_mutual = False
        num_node = 'all'
        in_dim = 6
        inlier_threshold = 0.1
        downsample = 0.03
        cnt = 0

        # load ground truth transformation
        for scene in tqdm(scene_list, desc='process testing data'):
            scene_path = f'{root}/fragments/{scene}'
            gt_path = f'{root}/gt_result/{scene}-evaluation'
            gt_trans = {}
            for k, v in __loadlog__(gt_path).items():
                gt_trans[f'{scene}@{k}'] = v

            for idx in range(len(gt_trans)):
                test_result = {}
                # load meta data
                key = list(gt_trans.keys())[idx]
                scene = key.split('@')[0]
                src_id = key.split('@')[1].split('_')[0]
                tgt_id = key.split('@')[1].split('_')[1]

                # load point coordinates and pre-computed per-point local descriptors
                if descriptor == 'fcgf':
                    src_data = np.load(
                        f"{root}/fragments/{scene}/cloud_bin_{src_id}_fcgf.npz")
                    tgt_data = np.load(
                        f"{root}/fragments/{scene}/cloud_bin_{tgt_id}_fcgf.npz")
                    src_keypts = src_data['xyz']
                    tgt_keypts = tgt_data['xyz']
                    src_features = src_data['feature']
                    tgt_features = tgt_data['feature']
                elif descriptor == 'fpfh':
                    src_data = np.load(
                        f"{root}/fragments/{scene}/cloud_bin_{src_id}_fpfh.npz")
                    tgt_data = np.load(
                        f"{root}/fragments/{scene}/cloud_bin_{tgt_id}_fpfh.npz")
                    src_keypts = src_data['xyz']
                    tgt_keypts = tgt_data['xyz']
                    src_features = src_data['feature']
                    tgt_features = tgt_data['feature']
                    src_features = src_features / \
                        (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
                    tgt_features = tgt_features / \
                        (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)
                
                # compute ground truth transformation
                # orig_trans: the given ground truth trans is target-> source
                orig_trans = np.linalg.inv(gt_trans[key])
                # data augmentation
                aug_R = rotation_matrix(augment_axis, augment_rotation)
                aug_T = translation_matrix(augment_translation)
                aug_trans = integrate_trans(aug_R, aug_T)
                # Equation: trans_pts = R @ pts + t
                tgt_keypts = transform(tgt_keypts, aug_trans)
                # gt_pose: trans1 @ trans2
                gt_pose = concatenate(aug_trans, orig_trans)

                # select {num_node} numbers of keypoints
                N_src = src_features.shape[0]
                N_tgt = tgt_features.shape[0]
                # use all point during test.
                if num_node == 'all':
                    src_sel_ind = np.arange(N_src)
                    tgt_sel_ind = np.arange(N_tgt)
                else:
                    src_sel_ind = np.random.choice(N_src, num_node)
                    tgt_sel_ind = np.random.choice(N_tgt, num_node)
                src_desc = src_features[src_sel_ind, :]
                tgt_desc = tgt_features[tgt_sel_ind, :]
                src_keypts = src_keypts[src_sel_ind, :]
                tgt_keypts = tgt_keypts[tgt_sel_ind, :]

                # construct the correspondence set by mutual nn in feature space.
                distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
                source_idx = np.argmin(distance, axis=1)
                if use_mutual:
                    target_idx = np.argmin(distance, axis=0)
                    mutual_nearest = (target_idx[source_idx]
                                    == np.arange(source_idx.shape[0]))
                    corr = np.concatenate([np.where(mutual_nearest == 1)[
                        0][:, None], source_idx[mutual_nearest][:, None]], axis=-1)
                else:
                    corr = np.concatenate([np.arange(source_idx.shape[0])[
                        :, None], source_idx[:, None]], axis=-1)

                # build the ground truth label
                frag1 = src_keypts[corr[:, 0]]
                frag2 = tgt_keypts[corr[:, 1]]
                # transform: trans_pts = R @ pts + t
                frag1_warp = transform(frag1, gt_pose)
                distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
                labels = (distance < inlier_threshold).astype(np.int32)


                test_result['file_0'] = scene + '_' + src_id
                test_result['file_1'] = scene + '_' + tgt_id
                test_result['xyz_0'] = src_keypts.astype(np.float32)
                test_result['xyz_1'] = tgt_keypts.astype(np.float32)
                test_result['feat_0'] = src_desc
                test_result['feat_1'] = tgt_desc
                test_result['gt_pose'] = gt_pose
                test_result['corr'] = corr
                test_result['labels'] = labels

                pcd_path = os.path.join(out_path, '%s.pkl' % cnt)
                with open(pcd_path, 'wb') as f:
                    pickle.dump(test_result, f, pickle.HIGHEST_PROTOCOL)

                cnt += 1
