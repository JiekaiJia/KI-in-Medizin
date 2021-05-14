""""""
#  -*- coding: utf-8 -*-
# date: 2021
# author: AllChooseC

import csv
import os
import zipfile

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset


def read_data(zip_path='../data/training.zip', data_path='../data/raw_data'):
    """
    :param data_path:
    :param zip_path:
    :return:
    """

    if not os.path.exists(data_path):
        zip_file = zipfile.ZipFile(zip_path)
        zip_list = zip_file.namelist()  # Return a list of file names in the archive
        for f in zip_list:
            zip_file.extract(f, data_path)  # extracted data to the specific folder

        zip_file.close()  # close the zip file

    print(f'the file was unzipped to {data_path}.')

    raw_data = []
    label = []
    with open(f'{data_path}/training/REFERENCE.csv') as csv_file:  # Read data and labels
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            data = sio.loadmat(f'{data_path}/training/{row[0]}.mat')  # Import ECG data
            raw_data.append(data['val'][0])
            label.append(row[1])
            line_count = line_count + 1
            if (line_count % 100) == 0:
                print(f'{line_count}\t data was read.')

    data_df = pd.DataFrame({'raw_data': raw_data, 'label': label})  # Write the raw data into dataframe

    return data_df


# display_df(data_df, random=True, layers=4)
def max_min_length(data_df):
    """

    :param data_df:
    :return:
    """
    num_signals = len(data_df)
    max_ = -1000000000
    min_ = 1000000000
    # search the max and min length of ECG signal
    for i in range(num_signals):
        n = len(data_df.iloc[i, 0])
        if n > max_:
            max_ = n
        if n < min_:
            min_ = n

    print(f'max_length: {max_}, min_length: {min_}')
    return max_, min_


def cut_signal(num_signals, data_df, wished_length):
    """

    :param num_signals:
    :param data_df:
    :param wished_length:
    :return:
    """
    data_list = []
    label_list = []
    # Divide each signal into several min_length size segments
    for i in range(num_signals):
        n = len(data_df.iloc[i, 0]) // wished_length
        rest = len(data_df.iloc[i, 0]) % wished_length
        for k in range(n):
            data_list.append(data_df.iloc[i, 0][:wished_length])
            label_list.append(data_df.iloc[i, 1])

    pre_data_df = pd.DataFrame({'pre_data': data_list, 'label': label_list})
    return pre_data_df


class EcgDataset(Dataset):
    """ECG dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the signals.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.ecg_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['N', 'A', 'O', '~']

    def __len__(self):
        return len(self.ecg_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        signal_name = os.path.join(self.root_dir, self.ecg_frame.iloc[idx, 0])
        tmp = sio.loadmat(f'{signal_name}.mat')  # Import ECG data
        signal = tmp['val'][0]
        signal = signal.reshape(1, -1)
        label = self.ecg_frame.iloc[idx, 1]
        target = self.classes.index(label)

        if self.transform:
            signal = self.transform(signal)

        return signal, target


