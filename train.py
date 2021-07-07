# /usr/bin/env python3
""""""
#  -*- coding: utf-8 -*-
# date: 2021
# author: AllChooseC

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from data_preprocessing import (
    EcgDataset,
    max_min_length,
    read_data,
)
from metrics import accuracy, f1_score_, report, f1_score_binary
from models import CnnBaseline, Cnn2018
from train_help_function import evaluate, fit
from transforms import (
    DropoutBursts,
    ToSpectrogram,
    RandomResample,
    Rescale,
    ToTensor,
    Normalize
)
from utils import (
    DeviceDataLoader,
    get_data_loader,
    get_default_device,
    load_model,
    to_device,
)
from sklearn.preprocessing import normalize
from visualization import display_signal,display_spectrogram


if __name__ == '__main__':
    # Read data as dataframe
    data_df = read_data(
        zip_path='training.zip',
        data_path='../training',
        # data_path='../test_examples',
    )
    num_signals = len(data_df)
    labels = data_df['label'].tolist()
    onehot_labels = np.zeros((num_signals, 4))
    for i, label in enumerate(labels):
        if label == 'N':
            onehot_labels[i][0] = 1
        elif label == 'A':
            onehot_labels[i][1] = 1
        elif label == 'O':
            onehot_labels[i][2] = 1
        else:
            onehot_labels[i][3] = 1
    # Compute the max and min signal length
    max_len, min_len = max_min_length(data_df)
    # Define the data transform
    basic = transforms.Compose([
        Normalize(),
        Rescale(max_len),
        ToSpectrogram(nperseg=64, noverlap=32),
        ToTensor()
    ])
    augment = transforms.Compose([
        Normalize(),
        DropoutBursts(threshold=2, depth=10),
        RandomResample(),
        Rescale(max_len),
        ToSpectrogram(nperseg=64, noverlap=32),
        ToTensor()
    ])
    composed = [augment, basic]
    # Create dataset for training
    train_dataset = EcgDataset(
        csv_file='../training/REFERENCE.csv',
        root_dir='../training',
        # csv_file='../test_examples/REFERENCE.csv',
        # root_dir='../test_examples',
        transform=composed,
    )
    vld_dataset = EcgDataset(
        csv_file='../training/REFERENCE.csv',
        root_dir='../training',
        # csv_file='../test_examples/REFERENCE.csv',
        # root_dir='../test_examples',
        transform=composed,
        train=False
    )
    # from torch.utils.data.dataloader import DataLoader
    # mm = DataLoader(vld_dataset, 1)
    # model = Cnn2018()
    # model = load_model(model, f'./models/cnn2018_paramsN_1.pth', evaluation=True)
    # # Print the best model's results
    # report(model,mm)
    #
    # import sys
    # sys.exit(1)

    # Split dataset into train dataset and test dataset
    compensation_factor = 0.2
    batch_size = 32
    train_loaders, vld_loaders = get_data_loader(train_dataset, vld_dataset, batch_size, onehot_labels, compensation_factor)
    # Get the default device 'cpu' or 'cuda'
    device = get_default_device()
    train_loaders_ = []
    vld_loaders_ = []
    # Create data loader iterator
    for train_loader, vld_loader in zip(train_loaders, vld_loaders):
        train_loaders_.append(DeviceDataLoader(train_loader, device))
        vld_loaders_.append(DeviceDataLoader(vld_loader, device))

    num_epochs = 500
    opt_fn = torch.optim.Adam
    lr = 1e-3
    # Train the model
    trial = 0
    train_with_pre = True
    for train_loader, vld_loader in zip(train_loaders_, vld_loaders_):
        trial += 1
        # Initialize the model
        # model = CnnBaseline()
        model = Cnn2018()
        if train_with_pre:
            if os.path.exists(f'./models/cnn2018_params2017_{trial}.pth'):
                model = load_model(model, f'./models/cnn2018_params2017_{trial}.pth', evaluation=False)
            elif os.path.exists(f'./models/cnn2018_params2017_{trial - 1}.pth'):
                model = load_model(model, f'./models/cnn2018_params2017_{trial - 1}.pth', evaluation=False)
            else:
                model = load_model(model, f'./models/cnn2018_params2017_multi5.pth', evaluation=False)
        else:
            if os.path.exists(f'./models/cnn2018_params2017_{trial - 1}.pth'):
                model = load_model(model, f'./models/cnn2018_params2017_{trial - 1}.pth', evaluation=False)
        # Transmit the model to the default device
        to_device(model, device)
        
        train_losses, train_metrics, vld_losses, vld_metrics = fit(
            num_epochs, trial, lr, model, F.cross_entropy,
            train_loader, vld_loader, f1_score_, opt_fn
        )

        # Load the best model
        model = Cnn2018()
        model = load_model(model, f'./models/cnn2018_paramsN_{trial}.pth', evaluation=True)
        to_device(model, device)
        # Print the best model's results
        report(model, vld_loader)