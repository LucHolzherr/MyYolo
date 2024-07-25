from tqdm import tqdm  # progress bar
import os
from datetime import datetime
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataset
import torchvision
# for visualisation of the model and the training
from torch.utils.tensorboard import SummaryWriter

from src.datasets.yolo_dataset import DataSetYolo
from src.metrics.mean_average_precision import mean_average_precision


@staticmethod   # TODO: REMOVE
def convert_from_my_to_cp(data):
    return [data.img_idx, data.class_pred, data.confidence, data.x_mid, data.y_mid, data.width, data.height]


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_dataset: DataSetYolo,
            val_dataset: DataSetYolo,
            loss_fn: nn.Module,
            optimizer,
            batch_size: int = 32,
            num_epochs: int = 10,
            is_tensorboard: bool = True,
            log_dir: str = 'runs',
            is_load_checkpoint: bool = False,
            is_save_checkpoint: bool = False,
            chkpoint_name_load: str = 'checkpoints/chkpoint',
            chkpoint_name_save: str = 'checkpoints/chkpoint',
            vis_conf_threshold: float = 0.5
    ):
        self.model = model
        self.num_classes = train_dataset.num_classes
        self.num_cells = train_dataset.S
        self.val_dataset = val_dataset
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = self.try_gpu()
        self.model.to(self.device)

        # checkpoints
        self.is_load_checkpoint = is_load_checkpoint
        self.is_save_checkpoint = is_save_checkpoint
        self.checkpoint_load_path = chkpoint_name_load
        self.checkpoint_save_path = chkpoint_name_save
        if self.is_save_checkpoint:
            # create directory if it does not exist
            directory = os.path.dirname(self.checkpoint_save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.vis_conf_threshold = vis_conf_threshold

        self.writer = None
        if is_tensorboard:
            # tensorboard, save run in folder of time
            now = datetime.now()
            # Format the datetime string
            formatted_datetime = now.strftime('%Y_%m_%d_%H_%M_%S')
            log_dir_run = f'{log_dir}/{formatted_datetime}'
            os.makedirs(log_dir_run)
            # create folder
            self.writer = SummaryWriter(log_dir_run)
            # randomly select 4 data-points from the validation set for which the prediction will be displayed
            # pred_dataset = self.val_loader.dataset
            pred_dataset = self.train_loader.dataset
            rnd_idxs = random.sample(
                range(len(pred_dataset)), min(4, len(pred_dataset)))
            # rnd_idxs = 0
            pred_subset = Subset(pred_dataset, rnd_idxs)
            self.pred_loader = DataLoader(
                pred_subset, batch_size=min(4, len(pred_dataset)), shuffle=False)

    def train_one_epoch(self, epoch, num_epochs, is_print_mAP: bool = False):
        self.model.train()
        running_loss = 0.0
        running_mAP = 0.0
        num_samples = len(self.train_loader.dataset)
        with tqdm(total=num_samples, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets, self.writer, epoch)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                # update progress bar
                pbar.set_postfix({'Training Loss': loss.item()})
                pbar.update(inputs.size(0))

                # print mAP
                if is_print_mAP:
                    mAP = self.compute_mean_average_precision(
                        outputs, targets)
                running_mAP += mAP

        epoch_loss = running_loss / len(self.train_loader)
        mAP_loss = running_mAP / len(self.train_loader)
        return epoch_loss, mAP_loss

    def validate_one_epoch(self, epoch, num_epochs, is_print_mAP: bool = True):
        self.model.eval()
        running_loss = 0.0
        running_mAP = 0.0
        num_samples = len(self.val_loader.dataset)
        with tqdm(total=num_samples, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    # print('next validation batch')
                    inputs, targets = inputs.to(
                        self.device), targets.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)

                    running_loss += loss.item()
                    # update progress bar
                    pbar.set_postfix({'Validation Loss': loss.item()})
                    pbar.update(inputs.size(0))

                    # print mean average precision
                    if is_print_mAP:
                        mAP = self.compute_mean_average_precision(
                            outputs, targets)
                        running_mAP += mAP

        epoch_loss = running_loss / len(self.val_loader)
        mAP_loss = running_mAP / len(self.train_loader)
        return epoch_loss, mAP_loss

    def _get_predicted_images(self, imgs, conf_threshold: float):
        imgs = imgs.to(self.device)
        # labels.to(self.device)
        outputs = self.model(imgs)
        # labels = labels.to(self.device)
        num_samples = imgs.shape[0]
        imgs_out = torch.zeros(
            (num_samples, imgs.shape[2], imgs.shape[3], imgs.shape[1]))
        for idx in range(num_samples):
            img_out = self.val_dataset.create_bbox_image_from_label_tensor(
                imgs[idx, ...], outputs[idx, ...], conf_threshold=conf_threshold)
            imgs_out[idx, ...] = img_out

        return imgs_out

    def compute_mean_average_precision(self, outputs, targets):
        bb_infos_gt = self.val_dataset.get_bb_info_from_tensor(targets)
        bb_infos_out = self.val_dataset.get_bb_info_from_tensor(outputs)
        mAP = mean_average_precision(
            bb_infos_out, bb_infos_gt)
        return mAP

    def train(self):
        # load checkpoint
        if self.is_load_checkpoint:
            start_epoch = self.load_checkpoint()
            if start_epoch >= self.num_epochs:
                raise ValueError(
                    f'Loaded checkpoint has already reached epoch {self.num_epochs}')
        else:
            start_epoch = 0

        best_val_loss = float('inf')
        for epoch in range(start_epoch, self.num_epochs):
            train_loss, train_mAP = self.train_one_epoch(
                epoch, self.num_epochs, is_print_mAP=True)
            val_loss, val_mAP = self.validate_one_epoch(
                epoch, self.num_epochs, is_print_mAP=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')

            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, train mAP: {train_mAP:.4f}")
            print(
                f"Validation Loss: {val_loss:.4f}, val. mAP: {val_mAP:.4f}\n")

            # save checkpoint
            if self.is_save_checkpoint:
                self.save_checkpoint(epoch + 1)

            # tensorboard visualisation
            if self.writer:
                # write the losses to tensorboard
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/mAP Train', train_mAP, epoch)
                self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.writer.add_scalar('Loss/mAp Validation', val_mAP, epoch)

                # predict images and display in tensorboard
                self.model.eval()
                pred_imgs, _ = next(iter(self.pred_loader))
                # pred_imgs, _ = next(iter(self.train_loader))
                pred_imgs = pred_imgs.to(self.device)
                pred_imgs_out = self._get_predicted_images(
                    pred_imgs, conf_threshold=self.vis_conf_threshold)
                pred_imgs_out = torch.permute(pred_imgs_out, (0, 3, 1, 2))
                # add to tensorboard
                grid = torchvision.utils.make_grid(pred_imgs_out)
                self.writer.add_image('images_val', grid, epoch)

                # # predict classes using the resnet
                # classes = self.model.evaluate_resnet18(pred_imgs)
                # if isinstance(classes, str):
                #     classes = [classes]
                # # print resnet classes as sanity check
                # print('resnet18 class prediction:')
                # for class_name in classes:
                #     print(f'\t{class_name}')

        if self.writer:
            self.writer.close()

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_load_path):
            checkpoint = torch.load(self.checkpoint_load_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            print(f"Checkpoint loaded from {self.checkpoint_load_path}")
            return epoch
        else:
            print(f"No checkpoint found at {self.checkpoint_load_path}")
            # Start from epoch 0 if no checkpoint is found
            return 0

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_save_path)
        print(f"Checkpoint saved at {self.checkpoint_save_path}")

    @ staticmethod
    def try_gpu():
        if torch.cuda.device_count() >= 1:
            return torch.device("cuda:0")
        return torch.device('cpu')
