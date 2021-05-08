import numpy as np
import warnings
warnings.filterwarnings("ignore")

from wettbewerb import load_references
from dataPreprocessing import preprocessing
from utils import build_dataloader, learn, plot
from classifierNet import ClassifierNet


if __name__ == '__main__':

    # set parameters
    PATH = '../training/'
    BATCH_SIZE = 2048
    EPOCH = 5
    SEG_LENGTH = 1000  # length of each segmented sample
    LR = 3e-4  # learning rate, optional: [1e-3, 3e-4]
    NUM = 5000  # the number of the training samples

    # load training data
    print("Loading data...")
    try:
        segments_with_labels = np.load('../segmented/segmented_train.npy')
    except:
        # load the dataset
        ecg_leads, ecg_labels, fs, ecg_names = load_references(PATH)
        ecg_leads, ecg_labels, ecg_names = ecg_leads[:NUM], ecg_labels[:NUM], ecg_names[:NUM]

        # pre-process the training samples
        segments_with_labels = preprocessing(ecg_leads=ecg_leads, ecg_labels=ecg_labels, fs=fs,
                                             length=SEG_LENGTH, data="train")
    X, y = segments_with_labels[:, :-1], segments_with_labels[:, -1][:, np.newaxis]
    del segments_with_labels

    # create data loader iterator
    train, test = build_dataloader(X, y, batch_size=BATCH_SIZE)
    del X, y

    # initialize a classification network
    classifier = ClassifierNet()

    # train the network
    loss, val_score = learn(classifier, train, test, lr=LR, epoch=EPOCH)

    # plot the loss values and validation scores
    plot(loss, val_score)



