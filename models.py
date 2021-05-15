"""This module provides the implemented models."""
#  -*- coding: utf-8 -*-
# date: 2021
# author: AllChooseC

import torch.nn as nn


class CnnBaseline(nn.Module):
    """Feedforward neural network with 8 hidden layer."""
    def __init__(self):
        super().__init__()
        # input 18286
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=2, padding=32),  # 9144
            # nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8),  # 1143

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=33, stride=2, padding=16),  # 573
            # nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=17, stride=2, padding=8),  # 288
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=2, padding=4),  # 145
            # nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5),  # 29

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),  # 16
            # nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=2, stride=1, padding=1),  # 16
            # nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),  # 4

            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=2, stride=1, padding=1),  # 4
            # nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),  # 4

            nn.Flatten(),
            nn.Linear(1024, 4)
        )

    def forward(self, train_x):
        return self.network(train_x)


def helper(i):
    """Helper function that computes the input of convolutional block."""
    return min(max(1, 32*i), 32) + 32 * i


def conv_layer(channels):
    """Basic convolutional layer unit."""
    return nn.Sequential(
        nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.15),
    )


def last_conv_layer(channels, out_channels):
    """The last layer of convolutional block that maxpool has padding."""
    return nn.Sequential(
        nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.15),
        nn.MaxPool2d(kernel_size=2, padding=1)
    )


def last_layer(channels, out_channels):
    """The last layer of convolutional block that maxpool has no padding."""
    return nn.Sequential(
        nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.15),
        nn.MaxPool2d(kernel_size=2)
    )


class ConvBlock(nn.Module):
    """Feedforward neural network with 4 hidden layer."""
    def __init__(self, channels=1, size=4):
        super().__init__()
        self.channels = channels
        self.size = size
        if self.channels == 1:
            self.out_channels = 64
        else:
            self.out_channels = self.channels + 32
        self.block = nn.ModuleList([conv_layer(self.channels) for _ in range(self.size-1)])
        self.block.append(last_conv_layer(self.channels, self.out_channels))

    def forward(self, train_x):
        for layer in self.block:
            train_x = layer(train_x)
        return train_x


class Cnn2018(nn.Module):
    """The module consists of 6 convolutional blocks, a average pooling, a full connection layer and a softmax layer."""
    def __init__(self):
        super().__init__()
        # 6 convolutional blocks
        self.network = nn.ModuleList([ConvBlock(helper(i), 4) for i in range(5)])
        self.network.extend([conv_layer(helper(5)) for _ in range(3)])
        self.network.append(last_layer(helper(5), helper(5)+32))
        # a average pooling that compute the average across time
        self.network.append(nn.AdaptiveAvgPool2d(1))
        # a full connection layer
        self.network.append(nn.Flatten())
        self.network.append(nn.Dropout(p=0.15))
        self.network.append(nn.Linear(224, 4))
        # a softmax layer
        self.network.append(nn.Softmax(dim=1))

    def forward(self, train_x):
        for layer in self.network:
            train_x = layer(train_x)
        return train_x


# from torchsummary import summary
#
# model = Cnn2018()
# # model = last_conv_layer(1,4)
# summary(model, input_size=(1, 280, 33), batch_size=-1)
