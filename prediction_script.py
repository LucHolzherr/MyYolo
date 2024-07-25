import torch
import torch.optim as optim
import random
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from src.datasets.vocdata import VOCDataSet
from src.datasets.kaggledata import KaggleDataSet
from src.datasets.yolo_dataset import DataSetYolo
from src.models.yolo_resnet import YoloResnet, YoloResnetAlternative

from src.losses.yolo_loss import YoloLoss

from src.utils import split_list_idxs
from src.metrics.mean_average_precision import mean_average_precision


NUM_LINEAR_LAYER = 1024
TRAIN_RESNET = True
TOTAL_DATASET_SIZE = 3000
TRAIN_VAL_RATIO = 0.85

VIS_CONF_THRESHOLD = 0.1
CHKPOINT_PATH = 'chkpoint_25_07_bt64_1024_kagle_02'


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Counter({'dog, 1': 2498, 'cat, 0': 1189})
if __name__ == "__main__":
    # load datasets
    # simple VOCdataset, label is a dictionary
    # dataset = VOCDataSet(max_num_samples=TOTAL_DATASET_SIZE)
    dataset = KaggleDataSet(max_num_samples=TOTAL_DATASET_SIZE)
    model = YoloResnet(num_linear_layer=NUM_LINEAR_LAYER,
                       train_resnet=TRAIN_RESNET,
                       num_classes=dataset.num_classes)
    # model = YoloResnetAlternative(train_resnet=TRAIN_RESNET)
    # load checkpoint
    checkpoint = torch.load(f'checkpoints/{CHKPOINT_PATH}')
    model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(
        f"number of trainable parameters: {count_trainable_parameters(model)}")

    # transform required for my model:
    pre_transform = model.pre_transform
    train_transform = v2.Compose([
        pre_transform,
    ])

    # train and validation datasets which return image and label in correct tensor format
    train_val_ratio = TRAIN_VAL_RATIO
    train_size = int(train_val_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_idxs, val_idxs = split_list_idxs(
        len(dataset), train_val_ratio, seed=2)
    train_dataset = DataSetYolo(
        dataset, train_idxs, input_shape=model.input_shape, transform=train_transform, inv_norm_transf=model.normalize_inv_transform)

    val_transform = v2.Compose([
        pre_transform,
        v2.Identity()]
    )
    val_dataset = DataSetYolo(
        dataset, val_idxs, input_shape=model.input_shape, transform=val_transform, inv_norm_transf=model.normalize_inv_transform)

    # define loss function
    loss_fn = YoloLoss(num_classes=dataset.num_classes,
                       class_counts={0: 1189, 1: 2498})
    # loss_fn = YoloLossCP()
    # random 10 idxs
    random_indices = random.choices(range(len(train_dataset)), k=30)
    mAP_avr = 0
    for idx in random_indices:
        img, label = train_dataset[idx]
        imgs = img.unsqueeze(0)
        labels = label.unsqueeze(0)
        # predict
        model.eval()
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        num_samples = imgs.shape[0]

        #
        imgs_bboxes = torch.zeros(
            (num_samples, imgs.shape[2], imgs.shape[3], imgs.shape[1]))
        for idx2 in range(num_samples):
            img_out = val_dataset.create_bbox_image_from_label_tensor(
                imgs[idx2, ...], outputs[idx2, ...], conf_threshold=VIS_CONF_THRESHOLD)
            imgs_bboxes[idx2, ...] = img_out

        bb_infos_gt = train_dataset.get_bb_info_from_tensor(labels)
        bb_infos_out = train_dataset.get_bb_info_from_tensor(outputs)
        mAP = mean_average_precision(
            bb_infos_out, bb_infos_gt)
        mAP_avr += mAP

        print(f"mAP: {mAP}")
        plt.figure(idx)
        plt.imshow(imgs_bboxes[0, ...])

    print(f"mAP avr: {mAP_avr/len(random_indices)}")

    plt.show()
    dbg_stop = True
