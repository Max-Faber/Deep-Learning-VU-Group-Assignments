import torch

def gen_batches(data, batch_size, device):
    batched_data_x = torch.tensor([])
    batched_data_y = torch.tensor([])
    n_samples = len(data)
    data_x = [data[d[1]][0] for d in data] # Extract the input values (X)
    data_y = [data[d[1]][1] for d in data] # Extract the output values (Y)
    for i in range(0, n_samples, batch_size):
        # Get batch out of the dataset and add them to the batches tensors
        batch_x = torch.stack(data_x[i:min(i + batch_size, n_samples)])
        batch_x = torch.stack([batch_x])
        batch_y = torch.tensor(data_y[i:min(i + batch_size, n_samples)])
        batch_y = torch.stack([batch_y])
        batched_data_x = torch.cat((batched_data_x, batch_x), 0)
        batched_data_y = torch.cat((batched_data_y, batch_y), 0)
    batched_data_x = batched_data_x.to(device)
    batched_data_y = batched_data_y.to(device)
    return batched_data_x, batched_data_y

def split_train_validation(training_data, batch_size, device, n_train=50000, n_validation=10000):
    # Split the data to train/test with a ratio of 50000:10000
    train, val = torch.utils.data.random_split(training_data, [n_train, n_validation])
    # Generate batches of the training data
    train_batches_x, train_batches_y = gen_batches(train, batch_size=batch_size, device=device)
    # Generate batches of the validation data
    val_batches_x, val_batches_y = gen_batches(val, batch_size=batch_size, device=device)
    return train_batches_x, train_batches_y, val_batches_x, val_batches_y
