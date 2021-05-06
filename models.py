""""""
#  -*- coding: utf-8 -*-
# date: 2021
# author: AllChooseC

import torch.nn as nn


class CnnBaseline(nn.Module):
    """Feedforward neural network with 1 hidden layer."""
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



