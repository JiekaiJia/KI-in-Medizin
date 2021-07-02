"""This module provides the basic functions about deep learning"""
#  -*- coding: utf-8 -*-
# date: 2021
# author: AllChooseC

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

class_distribution = [59.68, 8.68, 28.55, 3.08]
# 2017
class_distribution = [59.22, 8.65, 28.80, 3.33]


def split_indices(n, vld_pct, labels, compensation_factor, random_state=None):
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
    split_sets = [idxs[:n_vld], idxs[n_vld:2*n_vld], idxs[2*n_vld:3*n_vld], idxs[3*n_vld:4*n_vld], idxs[4*n_vld:]]
    train_sets = []
    vld_sets = []

    for k in range(5):
        train_set = np.concatenate((split_sets[k], split_sets[(k+1)%5], split_sets[(k+2)%5], split_sets[(k+3)%5]))
        masks = [labels[train_set, i].astype(bool) for i in range(labels.shape[1])]
        sets = [train_set[mask] for mask in masks]
        lst = []
        for idx, set_ in enumerate(sets):
            scale = int(100 * compensation_factor / class_distribution[idx]) + 1
            set_ = np.tile(set_, scale)
            set_ = set_.reshape([-1, 1])
            lst.append(set_)
        train_set = np.vstack(lst)
        train_set = train_set.squeeze()
        np.random.shuffle(train_set)
        
        train_sets.append(train_set)
        vld_sets.append(split_sets[k-1])
    
    if n_vld == 0:
        train_sets = []
        vld_sets = []
        train_set = idxs
        masks = [labels[:, i].astype(bool) for i in range(labels.shape[1])]
        sets = [train_set[mask] for mask in masks]
        lst = []
        for idx, set_ in enumerate(sets):
            scale = int(100 * compensation_factor / class_distribution[idx]) + 1
            set_ = np.tile(set_, scale)
            set_ = set_.reshape([-1, 1])
            lst.append(set_)
        train_set = np.vstack(lst)
        train_set = train_set.squeeze()
        np.random.shuffle(train_set)
        
        train_sets.append(train_set)
        vld_sets.append(idxs)

    return train_sets, vld_sets  # Pick the first n_vld indices for validation set


def get_data_loader(train_dataset, vld_dataset, batch_size, onehot_labels, compensation_factor):
    """This function generate the batch data for every epoch."""
    train_indices, vld_indices = split_indices(len(train_dataset), 0.2, onehot_labels, compensation_factor, random_state=2021)
    train_lds = []
    vld_lds = []
    for train_idx, vld_idx in zip(train_indices, vld_indices):
        train_sampler = SubsetRandomSampler(train_idx)
        train_ld = DataLoader(train_dataset, batch_size, sampler=train_sampler)
        vld_ld = DataLoader(vld_dataset, batch_size, sampler=vld_idx)
        train_lds.append(train_ld)
        vld_lds.append(vld_ld)

    return train_lds, vld_lds


def get_default_device():
    """Pick GPU if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensors to the chosen device."""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) if not isinstance(x, str) else x for x in data]
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
    device = get_default_device()
    all_preds = to_device(torch.tensor([]), device)
    all_labels = to_device(torch.tensor([]), device)
    all_names = []

    for batch in tqdm(loader):
        signals, labels = batch
        preds = model(signals)

        try:
            all_labels = torch.cat((all_labels, labels), dim=0)
        except TypeError:
            all_names.extend(labels)

        all_preds = torch.cat(
            (all_preds, preds)
            , dim=0
        )
    
    _, predicted = torch.max(all_preds, dim=1)

    if all_names:
      return predicted.cpu(), all_names

    return predicted.cpu(), all_labels.cpu()


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
        else:
            self.counter += 1
        print(f'INFO: Early stopping counter {self.counter} of {self.patience}')
        if self.counter >= self.patience:
            print('INFO: Early stopping')
            self.early_stop = True


def load_model(model, path, evaluation=False):
    """Load the saved model."""
    device = get_default_device()
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    if evaluation:
        # If the model is used to evaluation, the requires grad should be disabled.
        for parameter in model.parameters():
            parameter.requires_grad = False

    return model


device = get_default_device()


def get_length(data):
    # data shape [b, c, t, f]
    shape = list(data.shape)
    maps, _ = torch.max(torch.abs(data), 1)
    # data shape [b, t, f]
    used = torch.sign(maps)
    used = used.int()
    t_range = torch.arange(0, shape[2], device=device).unsqueeze(1)
    ranged = t_range * used
    length, _ = torch.max(ranged, 1)
    # data shape [b, f]
    length, _ = torch.max(length, 1)
    # data shape [b]
    length = length + 1
    return length


def set_zeros(data, length):
    shape = list(data.shape)
    # generate data shape matrix with time range with padding
    r = torch.arange(0, shape[1], device=device)
    r = torch.unsqueeze(r, 0)
    r = torch.unsqueeze(r, 2)
    r = r.repeat(shape[0], 1, shape[2])
    # generate data shape matrix with time range without padding
    l = torch.unsqueeze(length, 1)
    l = torch.unsqueeze(l, 2)
    l = l.repeat(1, shape[1], shape[2])
    # when col_n smaller than length mask entry is true
    mask = torch.lt(r, l)
    # when col_n larger than length, set input to zero
    output = torch.where(mask, data, torch.zeros_like(data))
    return output


def class_penalty(class_distribution, class_penalty=0.2):
    eq_w = [1 for _ in class_distribution]
    occ_w = [100/r for r in class_distribution]
    c = class_penalty
    weights = [[e * (1-c) + o * c for e,o in zip(eq_w, occ_w)]]
    class_weights = torch.Tensor(weights)

    return class_weights.to(device)
