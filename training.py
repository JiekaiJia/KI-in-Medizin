""""""
#  -*- coding: utf-8 -*-
# date: 2021
# author: AllChooseC

import numpy as np
import torch

from utils import EarlyStopping


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
    opt = opt(model.parameters(), lr=lr)

    early_stopping = EarlyStopping(patience=100, mode='max')
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

        early_stopping(vld_metric)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # print progress
        if metric is None:
            print(f'Epoch [{epoch+1}/{epochs}], train_loss: {train_loss:.4f}, validation_loss: {vld_loss:.4f}')
        else:
            print(f'Epoch [{epoch+1}/{epochs}], train_loss: {train_loss:.4f}, validation_loss: {vld_loss:.4f}, '
                  f'validation {metric.__name__}: {vld_metric:.4f}, train {metric.__name__}: {train_metric:.4f}')

    return train_losses, train_metrics, vld_losses, vld_metrics