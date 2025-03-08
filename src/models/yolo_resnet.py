import os
from collections import OrderedDict
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from torchvision.transforms import v2

from torch import nn
import torch.nn.functional as F
import numpy as np


class YoloResnet(nn.Module):
    """Uses the resnet18 network as base, and replaces the top layers with the ones from Yolo.
    """

    def __init__(self, num_linear_layer: int = 512, train_resnet: bool = False, num_cells: int = 7, num_boxes_per_cell: int = 2, num_classes: int = 20, **kwargs):
        super(YoloResnet, self).__init__(**kwargs)
        # parameters as member variables
        self.num_cells = num_cells
        self.num_boxes_per_cell = num_boxes_per_cell
        self.num_classes = num_classes

        # resnet model
        self.resnet_og = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # load classes
        with open("data/imagenet_classes.txt") as f:
            self.class_names = np.array([line.split(',')[1].strip()
                                         for line in f.readlines()])

        # remove the average pooling and the feed forward layer at the very end
        self.resnet_base = nn.Sequential(
            OrderedDict(list(self.resnet_og.named_children())[:-2]))
        # Freeze all layers # except for the last block
        if not train_resnet:
            for name, param in self.resnet_base.named_parameters():
                    param.requires_grad = False

        # 2 fully connected layers, input: 7*7*512, output = 1470 (=7*7*30)
        self.head = nn.Sequential(
            nn.Flatten(),
            # in paper its 4096, reduced to speedup training
            nn.Linear(in_features=7*7*512, out_features=num_linear_layer, bias=True),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.4),
            nn.Linear(
                in_features=num_linear_layer, 
                out_features=(self.num_classes + self.num_boxes_per_cell*5)*self.num_cells**2,
                bias=True),
        )

        # resnet transform which needs to be applied to input image (resizing, normalizing)
        self.input_shape = (224, 224)
        self.normalize_means = torch.tensor([0.485, 0.456, 0.406])
        self.normalize_stds = torch.tensor([0.229, 0.224, 0.225])
        self.pre_transform = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(self.input_shape[0]),
            v2.ToTensor(),
            v2.Normalize(mean=self.normalize_means, std=self.normalize_stds),
        ])
        self.normalize_inv_transform = v2.Normalize(
            mean=-self.normalize_means / self.normalize_stds, std=1./self.normalize_stds
        )

    def forward(self, x):
        x = self.resnet_base(x)
        x = self.head(x)
        return x.reshape(
            x.shape[0], self.num_boxes_per_cell * 5 + self.num_classes,
            self.num_cells, self.num_cells)

    def evaluate_resnet18(self, x):
        """Method evaluates the classification resnet18, which is part of the yolo architecture.
        """
        x = self.resnet_og(x)
        # softmax not needed, as it does not change the argmax
        x = torch.argmax(x, dim=1)
        class_name = self.class_names[x]
        return class_name


class YoloResnetAlternative(nn.Module):
    """Uses the resnet18 network as base, and replaces the top layers with 1x1 convolutions
    """

    def __init__(self, train_resnet: bool = False, num_cells: int = 7, num_boxes_per_cell: int = 2, num_classes: int = 20, **kwargs):
        super(YoloResnetAlternative, self).__init__(**kwargs)
        # parameters as member variables
        self.num_cells = num_cells
        self.num_boxes_per_cell = num_boxes_per_cell
        self.num_classes = num_classes

        # load classes
        with open("data/imagenet_classes.txt") as f:
            self.class_names = np.array([line.split(',')[1].strip()
                                         for line in f.readlines()])

        # resnet model
        self.resnet18_og = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # remove the average pooling and the feed forward layer at the very end
        self.resnet_base = nn.Sequential(
            OrderedDict(list(self.resnet18_og.named_children())[:-2]))

        # Freeze all layers # except for the last block
        if not train_resnet:
            for name, param in self.resnet_base.named_parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            self.get_conv_layer(out_channels=self.num_classes +
                                5*self.num_boxes_per_cell, kernel_size=3, padding=1),
            nn.LazyConv2d(out_channels=self.num_classes +
                          5*self.num_boxes_per_cell, kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=self.num_classes +
                          5*self.num_boxes_per_cell, kernel_size=1)
        )

        # resnet transform which needs to be applied to input image (resizing, normalizing)
        self.input_shape = (224, 224)
        self.normalize_means = torch.tensor([0.485, 0.456, 0.406])
        self.normalize_stds = torch.tensor([0.229, 0.224, 0.225])
        self.pre_transform = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(self.input_shape[0]),
            v2.ToTensor(),
            v2.Normalize(mean=self.normalize_means, std=self.normalize_stds),
        ])
        self.normalize_inv_transform = v2.Normalize(
            mean=-self.normalize_means / self.normalize_stds, std=1./self.normalize_stds
        )

    def get_conv_layer(
            self, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        return nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size, stride, padding),
            nn.LazyBatchNorm2d(),
            nn.ReLU())

    def forward(self, x):
        x = self.resnet_base(x)
        x = self.head(x)
        return x

    def evaluate_resnet18(self, x):
        """Method evaluates the classification resnet18, which is part of the yolo architecture.
        """
        x = self.resnet18_og(x)
        # softmax not needed, as it does not change the argmax
        x = torch.argmax(x, dim=1)
        class_name = self.class_names[x]
        return class_name


class Resnet(nn.Module):
    """Class for testing whether the resnet18 works on this dataset

    """

    def __init__(self, **kwargs):
        super(Resnet, self).__init__(**kwargs)

        # resnet model
        # weights=ResNet18_Weights.DEFAULT)
        self.resnet18 = models.resnet18(pretrained=True)

    def forward(self, x):
        x = self.resnet18(x)
        class_idx = torch.argmax(x, dim=1)
        return class_idx
