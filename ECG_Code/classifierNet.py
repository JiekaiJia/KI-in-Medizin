from torch import nn


class BiLSTM(nn.Module):
    def __init__(self, input_size=1000, hidden_size=100, output_size=4,
                 n_layers=2, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_directions = 2 if bidirectional else 1

        self.lstm1 = nn.LSTM(input_size=self.input_size,
                             hidden_size=self.hidden_size,
                             num_layers=n_layers,
                             dropout=.2,
                             batch_first=True,  # (batch, seq, feature)
                             bidirectional=bidirectional)
        self.linear = nn.Linear(self.hidden_size * self.n_directions, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)  # x: (batch, seq_len, num_directions * hidden_size)
        x = self.linear(x[:, -1, :])
        x = self.softmax(x)
        return x


# extract features of ecg-signals
class CNN1d(nn.Module):
    def __init__(self, output_size):
        super(CNN1d, self).__init__()  # (batch, 1, 1000)
        self.output_size = output_size

        self.conv1 = nn.Sequential(nn.Conv1d(1, 16, 50, 5),
                                   nn.BatchNorm1d(16),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2),
                                   nn.Dropout(p=0.1))  # (batch, 16, 95)
        self.conv2 = nn.Sequential(nn.Conv1d(16, 64, 45, 5),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2),
                                   nn.Dropout(p=0.1))  # (batch, 64, 5)
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, 3, 1),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.1))  # (batch, 128, 3)
        self.conv4 = nn.Sequential(nn.Conv1d(128, 256, 3, 1),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.1))  # (batch, 256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x.view(-1, 1, self.output_size)


class ClassifierNet(nn.Module):  # input -> CNN1d -> BiLSTM -> output
    def __init__(self):
        super(ClassifierNet, self).__init__()
        self.input_size = 256  # extract 256 features from each sample using CNN1d
        self.hidden_size = 100
        self.output_size = 4  # corresponding to 4 classes
        self.n_layers = 2
        self.cnn = CNN1d(self.input_size)
        self.lstm = BiLSTM(self.input_size, self.hidden_size, self.output_size, self.n_layers)

    def forward(self, x):
        features = self.cnn(x)
        output = self.lstm(features)
        return output