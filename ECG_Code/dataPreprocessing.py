"""
The functions of pre-processing data
"""
import numpy as np
import sys
from scipy import signal
from ecgdetectors import Detectors
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocessing(ecg_leads=None, ecg_labels=None, ecg_names=None, fs=300, length=1000, data="train"):
    """
    Pre-processes the samples.

    Parameters
    ----------
    ecg_leads : list of numpy arrays,
        ECG signal
    ecg_labels : list of str,
        labels corresponding to the ECG signals, 'N','A','O','~'
    ecg_names: list of str,
        names corresponding to the ECG signals, e.g. 'train_ecg_00001'
    fs : float,
        sampling frequency
    length: int,
        length of each segmented sample
    data : str,
        type of the dataset, 'train' or 'test'

    Returns
    -------
    segments_with_labels/names : numpy array,
        segmented ECG signals and their corresponding labels/numbers (in the last column)
    """

    detector = Detectors(fs)  # initialize the QRS-detector
    scaler = StandardScaler()  # initialize the standard scaler
    le = LabelEncoder()  # initialize the label encoder

    if data == "train":

        # encode the labels, 'A'->0, 'N'->1, 'O'->2, '~'->3
        ecg_labels = le.fit_transform(ecg_labels)
        ecg_labels = ecg_labels[:, np.newaxis]

        # segment the ecg-signals
        print("Segmenting data...")
        segments, labels = None, None
        for i, ecg_lead in enumerate(ecg_leads):
            print(i) ##########
            r_peaks = qrs_detection(ecg_lead, detector)  # calculate r-peaks

            # If there is a certain level of interference at the beginning of the signal,
            # the r-peaks cannot be calculated accurately.
            split_ecg_lead = ecg_lead
            while len(r_peaks) <= 6:
                split_ecg_lead = split_ecg_lead[900:]  # truncate the signal
                r_peaks = qrs_detection(split_ecg_lead, detector)  # recalculate r-peaks

            # segment signal base on the position of r-peaks
            seg, label = get_segments(ecg_lead=split_ecg_lead, ecg_label=ecg_labels[i],
                                      r_peaks=r_peaks, length=length, data=data)

            if segments is None:
                segments, labels = seg, label
            else:
                segments, labels = np.vstack((segments, seg)), np.vstack((labels, label))

        # pre-processing the ecg-signals
        print("Processing the segmented data...")
        segments_with_labels = np.hstack((signal_processing(segments, scaler), labels))

        # save the pre-processed data
        np.save('../segmented/segmented_train.npy', segments_with_labels)
        print(str(segments_with_labels.shape[0]), " segmented training data have been saved.")

        return segments_with_labels

    elif data == "test":

        print("Segmenting data...")
        segments, names = None, None
        for i, ecg_lead in enumerate(ecg_leads):
            print(i)  ##########
            # calculate r-peaks
            r_peaks = qrs_detection(ecg_lead, detector)
            split_ecg_lead = ecg_lead
            while len(r_peaks) <= 6:
                split_ecg_lead = split_ecg_lead[900:]
                r_peaks = qrs_detection(split_ecg_lead, detector)

            # segment signal
            seg, name = get_segments(ecg_lead=split_ecg_lead, ecg_name=ecg_names[i],
                                     r_peaks=r_peaks, length=length, data=data)

            if segments is None:
                segments, names = seg, name
            else:
                segments, names = np.vstack((segments, seg)), np.vstack((names, name))

        # pre-processing
        print("Processing the segmented data...")
        segments_with_names = np.hstack((signal_processing(segments, scaler), names))

        # save the pre-processed data
        np.save('../segmented/segmented_test.npy', segments_with_names)
        print(str(segments_with_names.shape[0]), " segmented test data have been saved.")

        return segments_with_names

    else:
        print("The type of dataset is wrong!")
        sys.exit()


def qrs_detection(ecg_lead, detector):
    """
    Calculates the r-peaks of signals.
    """
    r_peaks = detector.pan_tompkins_detector(ecg_lead)
    return r_peaks


def get_segments(ecg_lead=None, ecg_label=None, ecg_name=None, r_peaks=None, length=1000, data='train'):
    """
    Segments the signal depending on its r-peaks.
    """

    # segment the signal based on the given indices
    def segment(indices, r_peaks, ecg_lead, length=1000):
        segments = []
        for i in indices:
            front, rear = r_peaks[i], r_peaks[i + 3]
            padding = length - rear + front
            if padding % 2 == 0:
                front_padding = rear_padding = int(padding / 2)
            else:
                front_padding = int((padding - 1) / 2)
                rear_padding = int((padding + 1) / 2)

            if front_padding > front:
                rear_padding += front_padding - front
                front_padding = front
            if rear + rear_padding >= ecg_lead.shape[0]:
                rear_padding = (ecg_lead.shape[0] - 1) - rear
                front_padding = front - (ecg_lead.shape[0] - 1 - length)

            segments.append(ecg_lead[front - front_padding:rear + rear_padding].copy())

        return segments

    n = len(r_peaks)

    if data == 'train':
        if ecg_label in [0., 3.]:         # label = 'A' or '~'
            indices = range(2, n - 4, 1)
        elif ecg_label in [2.]:           # label = 'O'
            indices = range(2, n - 4, 2)
        else:                             # label = 'N'
            indices = range(2, n - 4, 3)

        segments = segment(indices, r_peaks, ecg_lead, length=length)
        segments = np.array(segments)

        num_of_seg = segments.shape[0]
        labels = np.reshape(np.repeat(ecg_label, num_of_seg), (num_of_seg, -1))
        return segments, labels

    else:
        indices = range(2, n - 4)
        segments = segment(indices, r_peaks, ecg_lead, length=length)
        segments = np.array(segments)

        num_of_seg = segments.shape[0]
        names = np.reshape(np.repeat(int(ecg_name[-5:]), num_of_seg), (num_of_seg, -1))
        return segments, names


def signal_processing(ecg_leads, scaler):
    """
    Standardizes and de-noises the signals.
    """
    ecg_leads = scaler.fit_transform(ecg_leads)

    b_1, a_1 = signal.butter(N=8, Wn=0.01, btype='lowpass')  # filter_1
    b_2, a_2 = signal.butter(N=8, Wn=0.15, btype='lowpass')  # filter_2

    ecg_filtered = []
    for ecg_lead in ecg_leads:
        curve = signal.filtfilt(b_1, a_1, ecg_lead)
        ecg_fixed = ecg_lead - curve  # deal with baseline drift
        ecg_filtered.append(signal.filtfilt(b_2, a_2, ecg_fixed))  # de-noise

    return ecg_filtered