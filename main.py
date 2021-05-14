# /usr/bin/env python3
""""""
#  -*- coding: utf-8 -*-
# date: 2021
# author: AllChooseC

from ecgdetectors import Detectors
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn.functional as F
from torchvision import transforms

from data_preprocessing import (
    EcgDataset,
    max_min_length,
    read_data,
)
from models import CnnBaseline
from transform import (
    ToSpectrogram,
    Rescale,
    ToTensor
)
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
    data_df = read_data(zip_path='../data/training.zip', data_path='../data/raw_data')
    num_signals = len(data_df)
    # Compute the max and min signal length
    max_len, min_len = max_min_length(data_df)
    # Define the data transform
    composed = transforms.Compose([Rescale(max_len), ToSpectrogram(nperseg=64, noverlap=32), ToTensor()])
    # Create dataset for training
    dataset = EcgDataset(
        csv_file='../data/raw_data/training/REFERENCE.csv',
        root_dir='../data/raw_data/training',
        transform=composed,
    )
    # Split dataset into train dataset and test dataset
    train_loader, vld_loader = get_data_loader(dataset, 32)
    # Get the default device 'cpu' or 'cuda'
    device = get_default_device()
    # Create data loader iterator
    train_loader = DeviceDataLoader(train_loader, device)
    vld_loader = DeviceDataLoader(vld_loader, device)
    # Initialize the model
    # model = CnnBaseline()
    # # Transmit the model to the default device
    # to_device(model, device)
    #
    # num_epochs = 10
    # opt_fn = torch.optim.Adam
    # lr = 1e-4
    # # Train the model
    # train_losses, train_metrics, vld_losses, vld_metrics = fit(
    #     num_epochs, lr, model, F.cross_entropy,
    #     train_loader, vld_loader, accuracy, opt_fn
    # )
    #
    # vld_loss, total, vld_acc = evaluate(model, F.cross_entropy, vld_loader, metric=accuracy)
    # _, _, train_acc = evaluate(model, F.cross_entropy, train_loader, metric=accuracy)
    # print(f'Loss: {vld_loss:.4f}, Accuracy: {vld_acc:.4f}')
    #
    # # Replace these values with your results
    # vld_accuracies = vld_metrics
    # train_accuracies = train_metrics
    # plt.plot(vld_accuracies, '-x')
    # plt.plot(train_accuracies, '-o')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.title('Accuracy vs. No. of epochs')
    # plt.show()
    #
    # # save model parameters
    # torch.save(model.state_dict(), './models/cnn1d_params.pkl')

