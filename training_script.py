from torch.utils.data import Dataset, random_split
import torch.optim as optim
from torchvision.transforms import v2

from src.datasets.vocdata import VOCDataSet
from src.datasets.kaggledata import KaggleDataSet
from src.datasets.yolo_dataset import DataSetYolo
from src.models.yolo_model import MyYolo
from src.models.yolo_resnet import YoloResnet, YoloResnetAlternative
from src.losses.yolo_loss import YoloLoss
from src.trainers.yolo_trainer import Trainer
from src.utils import split_list_idxs


# Hyperparameters etc.
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
WEIGHT_DECAY = 0.00
NUM_EPOCHS = 100
NUM_LINEAR_LAYER = 512
TOTAL_DATASET_SIZE = None
TRAIN_VAL_RATIO = 0.8
TRAIN_RESNET = True
LOAD_CHECKPOINT = False
SAVE_CHECKPOINT = True
CHKPOINT_LOAD = 'checkpoints/chkpoint_22_07_bt8_100_01_softmax_eucl'
CHPOINT_SAVE = 'checkpoints/chkpoint_25_07_bt16_100_01_noresnet'
VIS_CONF_THRESHOLD = 0.2


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # specify load datasets
    # VOCdataset, label is a dictionary
    dataset = VOCDataSet(max_num_samples=TOTAL_DATASET_SIZE)
    #dataset = KaggleDataSet(max_num_samples=TOTAL_DATASET_SIZE)

    # define model
    model = YoloResnet(num_linear_layer=NUM_LINEAR_LAYER,
                       train_resnet=TRAIN_RESNET,
                       num_classes=dataset.num_classes)
    print(
        f"number of trainable parameters: {count_trainable_parameters(model) // 10**6}M")
    # model = YoloResnetAlternative(train_resnet=TRAIN_RESNET)
    # model = Yolov1(in_channels=3, split_size=7, num_boxes=2, num_classes=20)

    # transform required for my model:
    pre_transform = model.pre_transform
    train_transform = v2.Compose([
        pre_transform,
        # v2.RandomResizedCrop(
        #    size=(224, 224), scale=(0.9, 1.0), antialias=True),
        v2.RandomHorizontalFlip()
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

    # define the optimizer
    lr_head = LEARNING_RATE
    if TRAIN_RESNET:
        param_dict = [{'params': model.resnet_base.parameters(),
                      'lr': 0.1 * lr_head},
                      {'params': model.head.parameters(), 'lr': lr_head}]
    else:
        param_dict = [{'params': model.head.parameters(), 'lr': lr_head}]
    optimizer = optim.Adam(param_dict,
                           weight_decay=WEIGHT_DECAY,
                           )

    # ExponentialLR scheduler (gamma=0.95 means lr reduces by 5% per epoch)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    
    # define the trainer
    trainer = Trainer(model, train_dataset, val_dataset,
                      loss_fn, optimizer, scheduler,
                      batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
                      is_tensorboard=True,
                      is_load_checkpoint=LOAD_CHECKPOINT, is_save_checkpoint=SAVE_CHECKPOINT,
                      chkpoint_name_load=CHKPOINT_LOAD, chkpoint_name_save=CHPOINT_SAVE,
                      vis_conf_threshold=VIS_CONF_THRESHOLD)

    trainer.train()
