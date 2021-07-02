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


def f1_score_binary(outputs, labels):
    TP = 0  # Richtig Positive
    TN = 0  # Richtig Negative
    FP = 0  # Falsch Positive
    FN = 0  # Falsch Negative

    _, preds = torch.max(outputs, dim=1)
    preds = preds.cpu().numpy().tolist()
    labels = labels.cpu().numpy().tolist()
    for label, pred in zip(labels, preds):
        if label == 1 and pred == 1:
            TP = TP + 1
        if label == 0 and pred != 1:
            TN = TN + 1
        if label == 0 and pred == 1:
            FP = FP + 1
        if label == 1 and pred != 1:
            FN = FN + 1
    
    try:
        F1 = TP / (TP + 1/2*(FP+FN))
    except ZeroDivisionError:
        return 1
    return F1


def report(model, vld_loader):
    model.eval()
    with torch.no_grad():
        preds, targets = get_all_preds(model, vld_loader)

    print('Classification report :')
    print(classification_report(targets, preds))
    print('Confusion matrix:')
    print(confusion_matrix(targets, preds))

