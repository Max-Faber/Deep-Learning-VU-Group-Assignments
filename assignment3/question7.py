import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader

def split_train_validation(training_data, batch_size, n_train=50000, n_validation=10000):
    # Split the data to train/test with a ratio of 50000:10000
    train, val = torch.utils.data.random_split(training_data, [n_train, n_validation])
    # Generate batches of the training data
    train_batches, val_batches = DataLoader(train, batch_size=batch_size), DataLoader(val, batch_size=batch_size)
    return train_batches, val_batches
