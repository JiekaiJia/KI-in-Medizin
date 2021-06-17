# /usr/bin/env python3
""""""
#  -*- coding: utf-8 -*-
# date: 2021
# author: AllChooseC

from ecgdetectors import Detectors
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
from metrics import accuracy, f1_score_, report
from models import CnnBaseline, Cnn2018
from training import evaluate, fit
from transforms import (
    DropoutBursts,
    ToSpectrogram,
    RandomResample,
    Rescale,
    ToTensor
)
from utils import (
    DeviceDataLoader,
    get_data_loader,
    get_default_device,
    load_model,
    to_device,
split_indices
)
from visualization import display_signal


if __name__ == '__main__':
    # Read data as dataframe
    data_df = read_data(
        zip_path='../data/training.zip',
        data_path='../data/raw_data'
        # zip_path='./training.zip',
        # data_path='./'
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
    composed = transforms.Compose([
        # DropoutBursts(threshold=2, depth=8),
        # RandomResample(),
        Rescale(max_len),
        ToSpectrogram(nperseg=64, noverlap=32),
        ToTensor()
    ])
    # Create dataset for training
    dataset = EcgDataset(
        # csv_file='../data/raw_data/training/REFERENCE.csv',
        # root_dir='../data/raw_data/training',
        csv_file='./training/REFERENCE.csv',
        root_dir='./training',
        transform=composed,
    )
    # Split dataset into train dataset and test dataset
    compensation_factor = 0.2
    batch_size = 20
    train_loader, vld_loader = get_data_loader(dataset, batch_size, onehot_labels, compensation_factor)
    # Get the default device 'cpu' or 'cuda'
    device = get_default_device()
    # Create data loader iterator
    train_loader = DeviceDataLoader(train_loader, device)
    vld_loader = DeviceDataLoader(vld_loader, device)
    # display_signal(train_loader)
    # Initialize the model
    # model = CnnBaseline()
    model = Cnn2018()
    # Transmit the model to the default device
    to_device(model, device)

    num_epochs = 500
    opt_fn = torch.optim.Adam
    lr = 1e-3
    # Train the model
    train_losses, train_metrics, vld_losses, vld_metrics = fit(
        num_epochs, lr, model, F.cross_entropy,
        train_loader, vld_loader, f1_score_, opt_fn
    )

    # Load the best model
    model = Cnn2018()
    model = load_model(model, evaluation=True)
    to_device(model, device)
    # Print the best model's results
    report(model, vld_loader)

    # Replace these values with your results
    vld_accuracies = vld_metrics
    train_accuracies = train_metrics
    plt.plot(vld_accuracies, '-x')
    plt.plot(train_accuracies, '-o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()
