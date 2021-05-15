"""This module provides the basic functions about deep learning"""
#  -*- coding: utf-8 -*-
# date: 2021
# author: AllChooseC

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def split_indices(n, vld_pct, random_state=None):
    """This function is used to split the data into train and validation.

    Args:
        n: the number of train data
        vld_pct: the percentage of validation data
        random_state: keep the random results same each time calling the function

    Returns:
        the indexes of 2 divided datasets(train indices, validation indices).
    """
    n_vld = int(vld_pct*n)  # Determine size of validation set
    if random_state:
        np.random.seed(random_state)  # Set the random seed(for reproducibility)
    idxs = np.random.permutation(n)  # Create random permutation of 0 to n-1

    return idxs[n_vld:], idxs[:n_vld]  # Pick the first n_vld indices for validation set


def get_data_loader(data_set, batch_size):
    """This function generate the batch data for every epoch."""
    train_indices, vld_indices = split_indices(len(data_set), 0.2, random_state=2021)
    train_sampler = SubsetRandomSampler(train_indices)
    train_ld = DataLoader(data_set, batch_size, sampler=train_sampler)
    vld_ld = DataLoader(data_set, batch_size)

    return train_ld, vld_ld


def get_default_device():
    """Pick GPU if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensors to the chosen device."""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a data loader to move data to a device."""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a data batch after moving it to device."""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches."""
        return len(self.dl)


@torch.no_grad()
def get_all_preds(model, loader):
    """Output model's predictions and targets.

    :param model:
    :param loader:
    :return:
    """
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    for batch in loader:
        signals, labels = batch

        preds = model(signals)
        all_labels = torch.cat(
            (all_labels, labels)
            , dim=0
        )
        all_preds = torch.cat(
            (all_preds, preds)
            , dim=0
        )
        _, predicted = torch.max(all_preds, dim=1)
    return predicted, all_labels


class EarlyStopping:
    """Early stopping to stop the training when the loss does not improve after certain epochs."""
    def __init__(self, patience=100, mode='max'):
        """
        :param patience: how many epochs to wait before stopping when loss is not improving
        """
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def __call__(self, val_metric):
        if self.best_metric is None:
            self.best_metric = val_metric
        elif self.best_metric > val_metric:
            if self.mode == 'max':
                self.counter += 1
            else:
                self.best_metric = val_metric
        elif self.best_metric < val_metric:
            if self.mode == 'max':
                self.best_metric = val_metric
            else:
                self.counter += 1
        print(f'INFO: Early stopping counter {self.counter} of {self.patience}')
        if self.counter >= self.patience:
            print('INFO: Early stopping')
            self.early_stop = True

