""""""
#  -*- coding: utf-8 -*-
# date: 2021
# author: AllChooseC

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy import signal
import torch

from data_preprocessing import read_data


class ToSpectrogram(object):
    """"""
    def __init__(self, nperseg=32, noverlap=16):
        assert isinstance(nperseg, int)
        assert isinstance(nperseg, int)
        self.nperseg = nperseg
        self.noverlap = noverlap

    def __call__(self, data):
        log_spectrogram = True
        fs = 300
        _, _, Sxx = signal.spectrogram(data, fs=fs, nperseg=self.nperseg, noverlap=self.noverlap)
        Sxx = np.transpose(Sxx, [0, 2, 1])
        if log_spectrogram:
            Sxx = abs(Sxx)
            mask = Sxx > 0
            Sxx[mask] = np.log(Sxx[mask])
        return Sxx


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_length (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
        """

    def __init__(self, output_length):
        assert isinstance(output_length, (int, tuple))
        self.output_size = output_length

    def __call__(self, signal):

        h = 1
        w = signal.shape[1]
        if isinstance(self.output_size, int):
            new_h = h
            new_w = self.output_size
        else:
            new_h, new_w = self.output_size

        if w < new_w:
            # If w is smaller, padding w with 0 util new_w
            signal = np.pad(signal, ((0, 0), (0, new_w-w)), 'constant', constant_values=(0, 0))
        elif w == new_w:
            pass
        else:
            # If w is larger, cut signal to length new_w
            signal = signal[:new_w]

        return signal


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, signal):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        signal = torch.from_numpy(signal)
        return signal.float()


def upscale(signals, upscale_factor=1):
    signals = np.repeat(signals, upscale_factor, axis=0)
    return signals


def random_resample(signals, upscale_factor=1):
    [n_signals,length] = signals.shape
    # pulse variation from 60 bpm to 120 bpm, expected 80 bpm
    new_length = np.random.randint(
        low=int(length*80/120),
        high=int(length*80/60),
        size=[n_signals, upscale_factor]
    )
    signals = [np.array(s) for s in signals.tolist()]
    new_length = [np.array(nl) for nl in new_length.tolist()]
    sigs = [stretch_squeeze(s,l) for s, nl in zip(signals,new_length) for l in nl]
    sigs = [fit_tolength(s, length) for s in sigs]
    sigs = np.array(sigs)
    return sigs


def random_resample_with_mean(signals, meanHRs):
    [n_signals,length] = signals.shape
    new_lengths = [np.random.randint(low=int(length*hr/120), high=int(length*hr/60)) for hr in meanHRs]
    signals = [np.array(s) for s in signals.tolist()]
    new_lengths = [np.array(nl) for nl in new_lengths]
    sigs = [stretch_squeeze(s,nl) for s,nl in zip(signals,new_lengths)]
    sigs = [fit_tolength(s, length) for s in sigs]
    sigs = np.array(sigs)
    return sigs


def resample_with_mean(signals, meanHRs):
    [n_signals,length] = signals.shape
    new_lengths = [int(length*hr/80) for hr in meanHRs]
    signals = [np.array(s) for s in signals.tolist()]
    new_lengths = [np.array(nl) for nl in new_lengths]
    sigs = [stretch_squeeze(s,nl) for s,nl in zip(signals,new_lengths)]
    sigs = [fit_tolength(s, length) for s in sigs]
    sigs = np.array(sigs)
    return sigs


def zero_filter(input, threshold=2, depth=8):
    shape = input.shape
    # compensate for lost length due to mask processing
    noise_shape = [shape[0], shape[1] + depth]
    noise = np.random.normal(0,1,noise_shape)
    mask = np.greater(noise, threshold)
    # grow a neighbourhood of True values with at least length depth+1
    for d in range(depth):
        mask = np.logical_or(mask[:, :-1], mask[:, 1:])
    output = np.where(mask, np.zeros(shape), input)
    return output


def stretch_squeeze(source, length):
    target = np.zeros([1, length])
    interpol_obj = sc.interpolate.interp1d(np.arange(source.size), source)
    grid = np.linspace(0, source.size - 1, target.size)
    result = interpol_obj(grid)
    return result


def fit_tolength(source, length):
    target = np.zeros([length])
    w_l = min(source.size, target.size)
    target[0:w_l] = source[0:w_l]
    return target


# data_df = read_data(zip_path='../data/training.zip', data_path='../data/raw_data')
# f, t, Sxx = signal.spectrogram(
#     data_df.iloc[200, 0],
#     fs=300,
#     nperseg=64,
#     noverlap=32
# )
# print(Sxx.shape)
# Sxx = np.transpose(Sxx, [0, 2, 1])
# print(Sxx)
# Sxx = abs(Sxx)
# mask = Sxx > 0
# Sxx[mask] = np.log(Sxx[mask])
# plt.pcolormesh(t, f, Sxx, shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()


