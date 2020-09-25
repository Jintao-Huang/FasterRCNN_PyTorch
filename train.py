# Author: Jintao Huang
# Time: 2020-6-7

import torch
# `models package` copy from torchvision-0.4.0
from models.detection import FasterRCNN
from models.detection.backbone_utils import resnet_fpn_backbone
from models.detection.rpn import AnchorGenerator
from utils.detection import Trainer, Logger, Tester, Checker, APCounter, Saver, LRScheduler, get_dataset_from_pickle
from tensorboardX import SummaryWriter

batch_size = 16
comment = "-fasterrcnn,wd=4e-5,bs=16,lr=0.05"

# --------------------------------
root_dir = r'.'
images_folder = 'JPEGImages'
pkl_folder = 'pkl'

train_pickle_fname = "images_targets_train_hflip.pkl"
test_pickle_fname = "images_targets_test.pkl"

labels_map = {
    0: "background",  # ignore
    1: "person",
    2: "car"
    # ...
}


# --------------------------------
def lr_func(epoch):
    if 0 <= epoch < 1:
        return 1e-4
    elif 1 <= epoch < 2:
        return 1e-3
    elif 2 <= epoch < 3:
        return 0.01
    elif 3 <= epoch < 4:
        return 0.025
    elif 4 <= epoch < 32:
        return 0.05
    elif 32 <= epoch < 38:
        return 0.02
    elif 38 <= epoch < 40:
        return 5e-3


def main():
    print("配置: %s" % comment, flush=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    backbone = resnet_fpn_backbone("resnet50", True)
    model = FasterRCNN(backbone, len(labels_map), max_size=1024, rpn_anchor_generator=AnchorGenerator(
        (16, 32, 64, 128, 256), (0.5, 1., 2.)  # (0.015, 0.1, 1.25, 3.)
    ), box_score_thresh=0.5, box_nms_thresh=0.5)

    optim = torch.optim.SGD(model.parameters(), 0, 0.9, weight_decay=4e-5)
    train_dataset = get_dataset_from_pickle(root_dir, train_pickle_fname, images_folder, pkl_folder)
    test_dataset = get_dataset_from_pickle(root_dir, test_pickle_fname, images_folder, pkl_folder)
    ap_counter = APCounter(labels_map, 0.5)
    writer = SummaryWriter(comment=comment)
    logger = Logger(50, writer)
    checker = Checker(Tester(model, train_dataset, batch_size, device, ap_counter, 1000),
                      Tester(model, test_dataset, batch_size, device, ap_counter, 1000),
                      Saver(model), logger, 5, 0)
    lr_scheduler = LRScheduler(optim, lr_func)
    trainer = Trainer(model, optim, train_dataset, batch_size, device, lr_scheduler, logger, checker)
    trainer.train((0, 40))
    writer.close()


if __name__ == "__main__":
    main()
