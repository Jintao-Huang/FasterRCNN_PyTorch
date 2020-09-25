# Author: Jintao Huang
# Time: 2020-5-21

import numpy as np
from utils.detection import XMLProcessor
import os

# --------------------------------
root_dir = os.path.expanduser(r'~/dataset/sonar')  # 数据集所在文件夹
images_folder = 'JPEGImages'
annos_folder = "Annotations"
pkl_folder = 'pkl'
pkl_fname = "images_targets.pkl"
train_pkl_fname = "images_targets_train_f.pkl"
# train_hflip_pkl_fname = "images_targets_train_hflip.pkl"
test_pkl_fname = "images_targets_test_f.pkl"

# FasterRCNN中有background类，所以从1开始计
category = {
    # -1 -> ignore
    "columnar": 1,
    "linear": 2,
    # ...
}
labels_map = {
    1: "columnar",
    2: "linear"
    # ...
}
colors_map = {  # bgr
    0: (0, 255, 0),
    1: (0, 0, 255)
    # ...
}
# --------------------------------
xml_processor = XMLProcessor(root_dir, images_folder, annos_folder, pkl_folder, category, labels_map, True)
xml_processor.xmls_to_pickle(pkl_fname)
xml_processor.split_train_test_from_pickle(pkl_fname, 1000, train_pkl_fname, test_pkl_fname)
# xml_processor.hflip_from_pickle(train_pkl_fname, train_hflip_pkl_fname)
xml_processor.calc_anchor_distribute(pkl_fname, np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
                                                          1, 1.25, 1.5, 1.75, 2, 3, 4, 8]),
                                     np.array([0, 8, 16, 24, 32, 64, 128, 256, 512, 1024]))
# xml_processor.show_dataset(train_hflip_pkl_fname, colors_map, True)
