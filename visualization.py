""""""
#  -*- coding: utf-8 -*-
# date: 2021
# author: AllChooseC

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from data_preprocessing import read_data

from transforms import DropoutBursts, RandomResample

matplotlib.use('TkAgg')


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


def plot_spectrogram(data):
    f, t, Sxx = signal.spectrogram(
        data.reshape(1, -1),
        fs=300,
        nperseg=64,
        noverlap=32
    )
    cmap = plt.get_cmap('jet')
    Sxx = abs(Sxx).squeeze()
    mask = Sxx > 0
    Sxx[mask] = np.log(Sxx[mask])
    plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap=cmap)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig('./figs/spectrogram.png', bbox_inches='tight', dpi=220)
    plt.show()


def plot_signal(tmp, tmp2, pic_name):
    t = (len(tmp) - 1) / 300
    t = np.linspace(0, t, len(tmp))
    plt.plot(t, tmp, label='origin')
    plt.plot(t, tmp2, label=pic_name)
    plt.xlim(10, 12)
    plt.ylabel('Potential [mV]')
    plt.xlabel('Time [sec]')
    plt.legend()
    plt.savefig(f'./figs/{pic_name}.png', bbox_inches='tight', dpi=220)
    plt.show()


if __name__ == '__main__':
    data_df = read_data(zip_path='../data/training.zip', data_path='../training')
    data = data_df.iloc[0, 0] / 1000
    data = data.reshape(1, -1)
    dropout = DropoutBursts(2, 10)
    random = RandomResample()
    data2 = dropout(data).squeeze()
    data3 = random(data).squeeze()
    data = data.squeeze()
    # plot_spectrogram(data)
    plot_signal(data, data2, 'DropoutBurst')
    plot_signal(data, data3, 'RandomResampling')

