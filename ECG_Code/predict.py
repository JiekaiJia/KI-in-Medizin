import numpy as np
import torch
from dataPreprocessing import preprocessing
from classifierNet import ClassifierNet

from wettbewerb import load_references


def predict_labels(ecg_leads=None, fs=None, ecg_names=None, use_pretrained=False):
    """
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.

    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    """

    ########################## 提交前删除 ##########################
    try:
        print('Loading test data...')
        segments_with_names = np.load('../segmented/segmented_test.npy')
    except:
        ecg_leads, _, fs, ecg_names = load_references()
        ecg_leads, ecg_names = ecg_leads[5000:], ecg_names[5000:]
        segments_with_names = preprocessing(ecg_leads=ecg_leads, ecg_names=ecg_names,
                                            fs=fs, length=1000, data="test")
    ########################## 提交前删除 ##########################
    # segments_with_names = preprocessing(ecg_leads=ecg_leads, ecg_names=ecg_names,
    #                                     fs=fs, length=1000, data="test")

    segments = segments_with_names[:, :-1].astype('float')
    names = segments_with_names[:, -1]
    inputs = torch.from_numpy(segments.reshape(-1, 1, 1000)).type(torch.FloatTensor)

    # load the pre-trained classification network
    print('Loading model...')
    PATH = None
    if use_pretrained:
        PATH = '../model/classifierNet_params.pkl'
    classifier = ClassifierNet()
    classifier.load_state_dict(torch.load(PATH))
    classifier.eval()

    # predict
    print('Predicting...')
    outputs = classifier(inputs)
    _, seg_preds = torch.max(outputs.data, dim=1)
    seg_preds = seg_preds.numpy()

    # combine the predicted results of sub-samples
    seg_predictions = list(zip(names.tolist(), seg_preds.tolist()))
    pred_dict = {}
    for name, pred in seg_predictions:
        name_str = "train_ecg_" + "0" * (5 - len(str(int(name)))) + str(int(name))
        if name_str in pred_dict:
            pred_dict[name_str].append(pred)
        else:
            pred_dict[name_str] = [pred]

    # get the final prediction
    label_dict = {1: 'N', 2: 'O', 3: '~', 0: 'A'}
    predictions = list()
    for i, name in enumerate(pred_dict.keys()):
        ### To be modified
        # The label with the highest number of occurrences is the final prediction.
        label = label_dict[max(pred_dict[name], key=pred_dict[name].count)]
        predictions.append((name, label))

    print(str(len(pred_dict.keys())) + "\t Dateien wurden verarbeitet.")
    return predictions  # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!