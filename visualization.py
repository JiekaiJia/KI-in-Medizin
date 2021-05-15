""""""
#  -*- coding: utf-8 -*-
# date: 2021
# author: AllChooseC

import matplotlib.pyplot as plt
import numpy as np


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


def display_signal(data_loader):
    """Display signals."""
    count = 0
    classes = ['N', 'A', 'O', '~']
    for xs, ys in data_loader:
        batch_size = xs.shape[0]
        xs = xs.numpy()
        ys = ys.numpy()
        plt.figure(figsize=(15, 10))
        for i in range(batch_size):
            if count < 4:
                count += 1
                ax = plt.subplot(2, 2, count)
                tmp = np.squeeze(xs[i])
                t = (len(tmp) - 1) / 300
                t = np.linspace(0, t, len(tmp))
                plt.plot(t, tmp)
                plt.xlabel('time/s')
                plt.ylabel('amplitude')
                plt.grid()
                ax.title.set_text(classes[ys[i]])
            else:
                count = 0
                plt.tight_layout()
                plt.show()
                plt.figure(figsize=(15, 10))
        break


def display_spectrogram(data_loader):
    """Display spectrogram."""
    count = 0
    classes = ['N', 'A', 'O', '~']
    for xs, ys in data_loader:
        batch_size = xs.shape[0]
        xs = xs.numpy()
        ys = ys.numpy()
        plt.figure(figsize=(15, 10))
        for i in range(batch_size):
            if count < 4:
                count += 1
                ax = plt.subplot(2, 2, count)
                tmp = np.squeeze(xs[i])
                t = (len(tmp) - 1) / 300
                t = np.linspace(0, t, len(tmp))
                plt.plot(t, tmp)
                plt.xlabel('time/s')
                plt.ylabel('amplitude')
                plt.grid()
                ax.title.set_text(classes[ys[i]])
            else:
                count = 0
                plt.tight_layout()
                plt.show()
                plt.figure(figsize=(15, 10))
        break
