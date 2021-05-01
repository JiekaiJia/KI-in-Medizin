# /usr/bin/env python3
""""""
#  -*- coding: utf-8 -*-
# date: 2021
# author: AllChooseC

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data_preprocessing import (
    cut_signal,
    read_data,
    max_min_length,
)
from models import CnnBaseline
from utils import (
    accuracy,
    DeviceDataLoader,
    evaluate,
    fit,
    get_data_loader,
    get_default_device,
    to_device
)


if __name__ == '__main__':
    # Read data as dataframe
    dataset = read_data(zip_path='../data/training.zip')
    num_signals = len(dataset)
    # Compute the max and min signal length
    max_len, min_len = max_min_length(dataset)
    # Divide each signal into several min_length size segments
    dataset = cut_signal(num_signals, dataset, min_len)
    num_signals = len(dataset)

    # Split dataset into train dataset and test dataset
    train_loader, vld_loader = get_data_loader(dataset, 64)
    # Get the default device 'cpu' or 'cuda'
    device = get_default_device()
    # Create data loader iterator
    train_loader = DeviceDataLoader(train_loader, device)
    vld_loader = DeviceDataLoader(vld_loader, device)
    # Initialize the model
    model = CnnBaseline()
    # Transmit the model to the default device
    to_device(model, device)

    num_epochs = 100
    opt_fn = torch.optim.Adam
    lr = 0.005
    # Train the model
    train_losses, vld_losses, metrics = fit(
        num_epochs, lr, model, F.cross_entropy,
        train_loader, vld_loader, accuracy, opt_fn
    )

    vld_loss, total, vld_acc = evaluate(model, F.cross_entropy, vld_loader, metric=accuracy)
    print(f'Loss: {vld_loss:.4f}, Accuracy: {vld_acc:.4f}')

    # Replace these values with your results
    accuracies = [vld_acc] + metrics
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()
    print(dataset['label'].unique())
