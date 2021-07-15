# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from data_preprocessing import EcgDataPredict
from models import Cnn2018
from transforms import (
    ToSpectrogram,
    Rescale,
    ToTensor,
    Normalize
)
from utils import (
    DeviceDataLoader,
    get_data_loader,
    get_default_device,
    load_model,
    to_device,
    get_all_preds,
)

###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads,fs,ecg_names,use_pretrained=False):
    '''
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
    '''

#------------------------------------------------------------------------------
# Euer Code ab hier
    batch_size = 32
    model_names = []
    for i in range(1):
        if os.path.exists('./models/cnn2018_paramsN_1.pth'):
            model_names.append('./models/cnn2018_paramsN_1.pth')
    classes = ['N', 'A', 'O', '~']
    device = get_default_device()
    if use_pretrained:
        for i in range(1):
            model_names.append('./models/cnn2018_paramsN_1.pth')
    # Load the best model
    models = [Cnn2018() for _ in range(1)]
    for idx, model in enumerate(models):
        model = load_model(model, model_names[idx], evaluation=True)
        model.eval()
        to_device(model, device)
    # models.reverse()

    num_signals = len(ecg_leads)
    max_len = 35400
    basic = transforms.Compose([
        Normalize(),
        Rescale(max_len),
        ToSpectrogram(nperseg=64, noverlap=32),
        ToTensor()
    ])
    test_dataset = EcgDataPredict(
        ecg_leads=ecg_leads,
        ecg_names=ecg_names,
        transform=basic,
    )
    test_dl = DataLoader(test_dataset, batch_size)
    test_dataloader = DeviceDataLoader(test_dl, device)
    label_idxs = []
    for model in models:
        label_idx, name = get_all_preds(model, test_dataloader)
        label_idx = label_idx.numpy()
        label_idxs.append(label_idx)
    
    label_idx = []
    for i in range(num_signals):
        m = [mm[i] for mm in label_idxs]
        label_idx.append(max(m,key=m.count)) 
    predictions = [(n, classes[idx]) for idx, n in zip(label_idx, name)]

#------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
