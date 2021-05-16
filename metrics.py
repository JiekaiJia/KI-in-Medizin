""""""
#  -*- coding: utf-8 -*-
# date: 2021
# author: AllChooseC

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)
import torch

from utils import get_all_preds


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)


def f1_score_(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return f1_score(labels.cpu(), preds.cpu(), average='weighted')


def report(model, vld_loader):
    with torch.no_grad():
        preds, targets = get_all_preds(model, vld_loader)

    print('Classification report :')
    print(classification_report(targets, preds))
    print('Confusion matrix:')
    print(confusion_matrix(targets, preds))
