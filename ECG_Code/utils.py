import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, f1_score


def build_dataloader(X, y, batch_size=64):
    """
    Creates a data loader iterator.
    """

    print("Building data loader...")
    df = pd.DataFrame(data=np.hstack((X, y)),
                      columns=[i for i in range(X.shape[1])] + ['Class'])

    y = df.Class
    X = df.drop('Class', axis=1)

    # divide into the training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.1, random_state=77)

    # convert data type to numpy array
    X_train, X_val = X_train.values, X_val.values
    y_train, y_val = y_train.values.astype('int'), y_val.values.astype('int')

    # convert data type to tensor
    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    X_val = torch.from_numpy(X_val).type(torch.FloatTensor)
    y_val = torch.from_numpy(y_val).type(torch.LongTensor)

    train = TensorDataset(X_train, y_train)
    val = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def learn(classifier, train, test, lr=1e-4, epoch=50):
    """
    Trains the network.
    """
    loss_data, val_score = [], [0.]

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    print("Starting training...")
    for i in range(epoch):
        i += 1
        for _, (minibatch_x, minibatch_y) in enumerate(train):
            minibatch_x = minibatch_x.view(-1, 1, 1000)
            output = classifier(minibatch_x)
            loss = loss_func(output, minibatch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print the score of the classifier at the i-th epoch
        y_pred, y_true = [], []
        for _, (minibatch_x, minibatch_y) in enumerate(test):
            minibatch_x = minibatch_x.view(-1, 1, 1000)
            minibatch_y = minibatch_y.numpy()
            outputs = classifier(minibatch_x)
            _, pred = torch.max(outputs.data, dim=1)
            pred = pred.numpy()
            y_pred += pred.tolist()
            y_true += minibatch_y.tolist()

        score = f1_score(y_true, y_pred, average='micro')
        if i % 1 == 0:
            print("Epoch: ", i,
                  " | Loss: %.3f" % loss.data.numpy(),
                  " | F1 Score: %.2f" % score)
            # print(metric(np.array(y_pred), np.array(y_true)))

        loss_data.append(loss.data.numpy())
        val_score.append(score)

    # save the trained network
    choice = input("Save or not?[y/n]")
    if choice == 'y':
        torch.save(classifier.state_dict(), '../model/classifierNet_params.pkl')

    return np.array(loss_data), np.array(val_score)


def metric(y_pred, y_true):
    return classification_report(y_true, y_pred, target_names=['A', 'N', 'O', '~'])


def plot(loss, val_score):
    plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(loss)), loss)
    plt.ylabel("Loss")
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(val_score)), val_score)
    plt.ylabel("Validation accuracy")
    plt.xlabel("Epoch")
    plt.show()
    print("The average validation score is: %.2f" % np.mean(val_score[1:]))

