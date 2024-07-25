import os
import torch
import torchvision
from torch import nn


class MyYolo(nn.Module):
    def __init__(self, num_cells: int = 7, num_boxes_per_cell: int = 2, num_classes: int = 20, **kwargs):
        super(MyYolo, self).__init__(**kwargs)
        # parameters as member variables
        self.num_cells = num_cells
        self.num_boxes_per_cell = num_boxes_per_cell
        self.num_classes = num_classes

        # self.tensor_out_shape = (self.num_cells, self.num_cells, self.num_boxes_per_cell * 5 + self.num_classes)

        # define the individual network blocks
        # input is: 448x448x3, output is: 112x112x64
        self.b1 = nn.Sequential(
            self.get_conv_layer(64, 7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2)
        )

        # input is: 112x112x64, outptu is: 56x56x192
        self.b2 = nn.Sequential(
            self.get_conv_layer(out_channels=192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )

        # input is: 56x56x192, output is: 28x28x512
        self.b3 = nn.Sequential(
            self.get_conv_layer(out_channels=128, kernel_size=1),
            self.get_conv_layer(out_channels=256, kernel_size=3, padding=1),
            self.get_conv_layer(out_channels=256, kernel_size=1),
            self.get_conv_layer(out_channels=512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )

        # input is: 28x28x512, output is: 14x14x1024
        conv_repeat4 = nn.Sequential(
            self.get_conv_layer(256, 1),
            self.get_conv_layer(512, 3, padding=1)
        )
        self.b4 = nn.Sequential(
            conv_repeat4,
            conv_repeat4,
            conv_repeat4,
            conv_repeat4,
            self.get_conv_layer(out_channels=512, kernel_size=1),
            self.get_conv_layer(out_channels=1024, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )

        # input is: 14x14x1024, output is: 7x7x1024
        conv_repeat2 = nn.Sequential(
            self.get_conv_layer(512, 1),
            self.get_conv_layer(1024, 3, padding=1)
        )
        self.b5 = nn.Sequential(
            conv_repeat2,
            conv_repeat2,
            self.get_conv_layer(1024, 3, padding=1),
            self.get_conv_layer(1024, 3, stride=2, padding=1)
        )

        # input is: 7x7x1024, output is 7x7x1024
        self.b6 = nn.Sequential(
            self.get_conv_layer(1024, 3, padding=1),
            self.get_conv_layer(1024, 3, padding=1)
        )

        # 2 fully connected layers, input: 7*7*1024, output = 1470 (=7*7*30)
        self.b7 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(out_features=1470),
            # nn.LeakyReLU(0.1),
            # nn.Dropout(p=0.5)
        )

        # entire net
        self.net = nn.Sequential(
            self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7
        )

    def forward(self, x):
        x = self.net(x).reshape(
            x.shape[0], self.num_boxes_per_cell * 5 + self.num_classes,
            self.num_cells, self.num_cells)
        return x

    def get_conv_layer(
            self, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        return nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size, stride, padding),
            nn.LazyBatchNorm2d(),
            nn.ReLU())
