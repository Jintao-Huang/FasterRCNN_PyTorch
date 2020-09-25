# Author: Jintao Huang
# Time: 2020-5-21


from utils.detection import XMLProcessor

# --------------------------------
root_dir = r'.'  # 数据集所在文件夹
images_folder = 'JPEGImages'
annos_folder = "Annotations"
pkl_folder = 'pkl'
pkl_fname = "images_targets.pkl"
train_pkl_fname = "images_targets_train.pkl"
train_hflip_pkl_fname = "images_targets_train_hflip.pkl"
test_pkl_fname = "images_targets_test.pkl"

category = {
    # 0 -> background / other
    "trash": 0,  # ignore
    "person": 1,
    "car": 2,
    # ...
}
labels_map = {
    0: "background",
    1: "person",
    2: "car"
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
xml_processor.hflip_from_pickle(train_pkl_fname, train_hflip_pkl_fname)
xml_processor.calc_anchor_distribute(pkl_fname)
xml_processor.show_dataset(train_hflip_pkl_fname, colors_map, True)
