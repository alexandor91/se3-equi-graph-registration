from dataloaders.ThreeDMatch import ThreeDMatchTest







if __name__ == '__main__':
    test_scene_list = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]

    all_stats = {}
    for scene_ind, scene in enumerate(test_scene_list):
        dset = ThreeDMatchTest(root='/import/network-temp/lzl_workspace/PointDSC/data/3DMatch',
                               descriptor=config.descriptor,
                               in_dim=config.in_dim,
                               inlier_threshold=config.inlier_threshold,
                               num_node='all',
                               use_mutual=config.use_mutual,
                               augment_axis=0,
                               augment_rotation=0.00,
                               augment_translation=0.0,
                               select_scene=scene,
                               )