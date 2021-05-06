"""This module provides the basic functions about deep learning"""
#  -*- coding: utf-8 -*-
# date: 2021
# author: AllChooseC

import matplotlib.pyplot as plt
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
    vld_sampler = SubsetRandomSampler(vld_indices)
    vld_ld = DataLoader(data_set, batch_size, sampler=vld_sampler)

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


def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    """Calculates the loss and metric value for a batch of data,
    and optionally performs gradient descent if an optimizer is provided."""
    preds = model(xb)
    loss = loss_func(preds, yb)

    if opt is not None:
        loss.backward()  # Compute gradients
        opt.step()  # Update parameters
        opt.zero_grad()  # Reset gradients

    metric_result = None
    if metric is not None:
        metric_result = metric(preds, yb)  # Compute the metric

    return loss.item(), len(xb), metric_result


def evaluate(model, loss_func, valid_dl, metric=None):
    """"""
    with torch.no_grad():
        # Pass each batch through the model
        results = [loss_batch(model, loss_func, xb, yb, metric=metric) for xb, yb in valid_dl]
        losses, nums, metrics = zip(*results)  # Separate losses, counts and metrics
        total = np.sum(nums)  # Total size of the dataset
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
    return avg_loss, total, avg_metric


def fit(epochs, lr, model, loss_func, train_dl, vld_dl, metric=None, opt=None):
    """"""
    train_losses, train_metrics, vld_losses, vld_metrics = [], [], [], []
    if opt is None:
        opt = torch.optim.SGD

    opt = opt(model.parameters(), lr=lr, weight_decay=0.05)
    for epoch in range(epochs):
        train_loss = 0
        train_metric = 0
        model.train()
        for xb, yb in train_dl:  # Training
            train_loss, _, train_metric = loss_batch(model, loss_func, xb, yb, opt, metric)

        model.eval()
        result = evaluate(model, loss_func, vld_dl, metric)  # Evaluation
        vld_loss, total, vld_metric = result
        vld_losses.append(vld_loss)  # Record the loss & metric
        vld_metrics.append(vld_metric)
        train_losses.append(train_loss)
        train_metrics.append(train_metric)

        # print progress
        if metric is None:
            print(f'Epoch [{epoch+1}/{epochs}], train_loss: {train_loss:.4f}, validation_loss: {vld_loss:.4f}')
        else:
            print(f'Epoch [{epoch+1}/{epochs}], train_loss: {train_loss:.4f}, validation_loss: {vld_loss:.4f}, '
                  f'validation {metric.__name__}: {vld_metric:.4f}, train {metric.__name__}: {train_metric:.4f}')

    return train_losses, train_metrics, vld_losses, vld_metrics


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)


def plot_metric(metric_values):
    """Plot metric values in a line graph."""
    plt.plot(metric_values, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')


def plot_losses(train_losses, vld_losses):
    plt.plot(train_losses, '-x')
    plt.plot(vld_losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')


def display_df(my_df, random=False, layers=4):
    for k in range(layers):
        plt.figure(figsize=(15, 10))
        for j in range(4):
            if random:
                idx = np.random.randint(0, len(my_df))
            else:
                idx = 4 * k + j
            case = my_df.iloc[idx, 1]
            plt.subplot(2, 2, j+1)
            plt.plot(my_df.iloc[idx, 0])
            plt.title(case)
        plt.show()
