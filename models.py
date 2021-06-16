"""This module provides the implemented models."""
#  -*- coding: utorch-8 -*-
# date: 2021
# author: AllChooseC

import torch
import torch.nn as nn
from utils import get_length, set_zeros


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


def conv2d_block(inputs_shape, length, kernel_size=5, out_channels=None, growth=32, depth=4, strides_end=2, max_pooling=1, drop_rate=0.15):

    # inherit layer-width from input
    in_channels = 1
    if strides_end is None:
        strides_end = 2
    if out_channels is None:
        in_channels = inputs_shape[1]
        out_channels = inputs_shape[1]

    conv = []
    strides = 1
    max_pooling_en = False

    for d in range(depth):
        if d != 0:
            in_channels = out_channels
        if d == depth-1:
            out_channels = out_channels + growth
            if max_pooling:
                max_pooling_en = True
            else:
                strides = strides_end

        conv.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=strides, padding=2))
        conv.append(nn.BatchNorm2d(out_channels))
        conv.append(nn.ReLU(inplace=True))
        if max_pooling_en:
            if max_pooling == 1:
                conv.append(nn.MaxPool2d(kernel_size=strides_end, padding=1))
            else:
                conv.append(nn.MaxPool2d(kernel_size=strides_end))
        # conv.append(nn.Dropout(p=drop_rate))

    block = nn.Sequential(*conv)

    length = torch.div((length + 1), 2)

    return block, length


class Cnn2018(nn.Module):

    def __init__(self):
        super().__init__()
        self.n_conv_blocks = 6
        self.out_channels_first = 32
        self.kernel_size = 5
        self.growth_block_end = 32
        self.strides_block_end = 2
        self.max_pooling = 2
        self.drop_rate = 0.15
        self.n_classes = 4

        self.linear1 = nn.Linear(224, self.n_classes)

    def forward(self, train_x):
        # Compute the original data length
        length = get_length(train_x)
        for i in range(self.n_conv_blocks):
            if i == 0:
                self.max_pooling = 2
                out_channels = self.out_channels_first
            else:
                out_channels = None
            if i == self.n_conv_blocks - 1:
                self.max_pooling = 1
            inputs_shape = list(train_x.shape)
            model, length = conv2d_block(
                    inputs_shape=inputs_shape,
                    length=length,
                    kernel_size=self.kernel_size,
                    out_channels=out_channels,
                    growth=self.growth_block_end,
                    strides_end=self.strides_block_end,
                    max_pooling=self.max_pooling,
                    drop_rate=self.drop_rate
            )
            train_x = model(train_x)

        [_, c_s, t_s, f_s] = list(train_x.shape)
        feature_seq = torch.reshape(train_x, [-1, t_s, f_s * c_s])
        # as we use affine functions our zero padded datasets
        # are now padded with the bias of the previous layers
        # in order to get the mean of only meaningful data out
        # set the zero-padding part back to zero again
        data = set_zeros(feature_seq, length)
        # as we have zero padded data,
        # reduce_mean would result into too small values for most sequences
        # therefore use reduce_sum and divide by actual length instead
        data = torch.sum(data, dim=1)
        length = torch.unsqueeze(length, 1)
        features = torch.div(data, length.float())

        preds = nn.Softmax(dim=1)(self.linear1(features))

        return preds

from torchsummary import summary
# model = Cnn2018()
# model = last_conv_layer(1,4)
# summary(model, input_size=(1, 570, 33), batch_size=-1)
