import numpy as np
import csv
from predict import predict_labels
from wettbewerb import save_predictions
from sklearn.metrics import classification_report


# #### 测试数据预处理功能
# signals_labels = np.load('../segmented/segmented_train.npy')
# print(signals_labels.shape)
# signals = signals_labels[:, :-1]
# labels = list(signals_labels[:, -1])
# print(signals[:5])
# print(labels[:5])

#### 预测不可见测试集的得分
predictions = predict_labels(use_pretrained=True)
save_predictions(predictions)
y_true, y_pred = [], []
with open('../training/REFERENCE.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        label = row[1]
        y_true.append(label)
y_true = y_true[5000:6000]
with open('PREDICTIONS.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        label = row[1]
        y_pred.append(label)
print(classification_report(y_true, y_pred, target_names=['A', 'N', 'O', '~']))

