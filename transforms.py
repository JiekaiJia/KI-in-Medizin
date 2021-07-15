""""""
#  -*- coding: utf-8 -*-
# date: 2021
# author: AllChooseC

import numpy as np
import scipy as sc
from scipy import signal
from sklearn.preprocessing import normalize
import torch


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
            # Data dimension [channels, times, frequency]
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

    def __call__(self, data):

        h = 1
        w = data.shape[1]
        if isinstance(self.output_size, int):
            new_h = h
            new_w = self.output_size
        else:
            new_h, new_w = self.output_size

        if w < new_w:
            # If w is smaller, padding w with 0 util new_w
            data = np.pad(data, ((0, 0), (0, new_w - w)), 'constant', constant_values=(0, 0))
        elif w == new_w:
            pass
        else:
            # If w is larger, cut signal to length new_w
            data = data[:, :new_w]

        return data


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        data = torch.from_numpy(data)
        return data.float()


class DropoutBursts(object):
    """Dropout bursts are created by selecting time instants uniformly at random and setting the ECG signal values in
    a 50ms vicinity of those time instants to 0. Dropout burst hence model short periods of weak signal due to, e.g.,
    bad contact of ECG leads.
    """

    def __init__(self, threshold=2, depth=8):
        self.threshold = threshold
        self.depth = depth

    def __call__(self, data):
        shape = data.shape
        # compensate for lost length due to mask processing
        noise_shape = [shape[0], shape[1] + self.depth]
        noise = np.random.normal(0, 1, noise_shape)
        mask = np.greater(noise, self.threshold)
        # grow a neighbourhood of True values with at least length depth+1
        for d in range(self.depth):
            mask = np.logical_or(mask[:, :-1], mask[:, 1:])
        output = np.where(mask, np.zeros(shape), data)
        output = output.reshape(1, -1)
        return output


def _stretch_squeeze(source, length):
    target = np.zeros([1, length])
    interpol_obj = sc.interpolate.interp1d(np.arange(source.size), source)
    grid = np.linspace(0, source.size - 1, target.size)
    result = interpol_obj(grid)
    return result


def _fit_to_length(source, length):
    target = np.zeros([length])
    w_l = min(source.size, target.size)
    target[0:w_l] = source[0, 0:w_l]
    return target


class RandomResample(object):
    """Assuming a heart rate of 80bpm for all training ECG signals, random resampling emulates a broader range of heart
    rates by uniformly resampling the ECG signals such that the heart rate of the resampled signal is uniformly
    distributed on the interval [60, 120]bpm.
    """

    def __call__(self, data):
        shape = data.shape
        # pulse variation from 60 bpm to 120 bpm, expected 80 bpm
        new_length = np.random.randint(
            low=int(shape[1] * 80 / 120),
            high=int(shape[1] * 80 / 60),
        )
        sig = _stretch_squeeze(data, new_length)
        sig = _fit_to_length(sig, shape[1])
        sig = sig.reshape(1, -1)
        return sig


class Normalize(object):
    """
    """
    def __call__(self, data):
        sig = normalize(data)*100
        mean_sig = np.mean(sig)
        sig = sig - mean_sig
        return sig

