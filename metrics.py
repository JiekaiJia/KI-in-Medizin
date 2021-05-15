""""""
#  -*- coding: utf-8 -*-
# date: 2021
# author: AllChooseC

import torch


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)
