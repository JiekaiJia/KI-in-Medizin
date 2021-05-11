import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn.functional as F
from torchvision import transforms

from data_preprocessing import (
    EcgDataset,
    max_min_length,
    read_data,
    Rescale,
    ToTensor,
)
from models import CnnBaseline
from utils import (
    accuracy,
    DeviceDataLoader,
    display_df,
    evaluate,
    fit,
    get_all_preds,
    get_data_loader,
    get_default_device,
    to_device
)


if __name__ == '__main__':
    # Read data as dataframe
    data_df = read_data(zip_path='../data/training.zip', data_path='../data/raw_data')
    display_df(data_df, random=True)
    # num_signals = len(data_df)
    # # Compute the max and min signal length
    # max_len, min_len = max_min_length(data_df)
    # # Define the data transform
    # composed = transforms.Compose([Rescale(max_len), ToTensor()])
    # # Create dataset for training
    # dataset = EcgDataset(
    #     csv_file='../data/raw_data/training/REFERENCE.csv',
    #     root_dir='../data/raw_data/training',
    #     transform=composed,
    # )
    # # Split dataset into train dataset and test dataset
    # train_loader, vld_loader = get_data_loader(dataset, 32)
    # # Get the default device 'cpu' or 'cuda'
    # device = get_default_device()
    # # Create data loader iterator
    # train_loader = DeviceDataLoader(train_loader, device)
    # vld_loader = DeviceDataLoader(vld_loader, device)
    # # Initialize the model
    # model = CnnBaseline()
    # # Transmit the model to the default device
    # to_device(model, device)
    # model.load_state_dict(torch.load('./models/cnn1d_params.pkl'))
    # # Results
    # with torch.no_grad():
    #     preds, targets = get_all_preds(model, vld_loader)
    #
    # print('Classification report :')
    # print(classification_report(targets, preds))
    # print('Confusion matrix:')
    # print(confusion_matrix(targets, preds))
