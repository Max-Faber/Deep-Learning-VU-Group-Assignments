import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time

from question7 import split_train_validation
from question8 import get_device
from question8 import get_batch_size
from datetime import datetime

class CNN_MNIST_N(nn.Module):
    def __init__(self, batch_size, device, N):
        super().__init__()
        self.batch_size = batch_size
        self.N = N

        self.relu = nn.ReLU()

        self.conv1_2d = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1, device=device)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2_2d = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, device=device)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3_2d = nn.Conv2d(in_channels=32, out_channels=N, kernel_size=(3, 3), stride=1, padding=1, device=device)
        self.max_pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.linear = nn.Linear(in_features=N, out_features=10, device=device)

    def forward(self, x, global_pooling):
        x = self.conv1_2d(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.conv2_2d(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        x = self.conv3_2d(x)
        x = self.relu(x)
        x = self.max_pool3(x)
        if global_pooling == 'mean':
            x = x.flatten(2, 3).mean(2)
        elif global_pooling == 'max':
            x = x.flatten(2, 3).amax(2)
        x = self.linear(x)
        return x

def cnn(train, val, batch_size, N, global_pooling_type, opt, learning_rate=0.001, momentum=None):
    epochs = 10
    device = get_device()

    cnn = CNN_MNIST_N(batch_size=batch_size, device=device, N=N)
    cnn.to(device)
    n_params = sum([p.numel() for p in cnn.parameters()])
    print(f'n_params: {n_params}')
    criterion = nn.CrossEntropyLoss()
    if opt == 'adam':
        optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=momentum)
    val_results = training_loop(epochs=epochs, train_dataset=train, val_dataset=val, cnn=cnn, criterion=criterion, optimizer=optimizer, batch_size=batch_size, device=device, global_pooling_type=global_pooling_type)
    return val_results

def training_loop(epochs, train_dataset, val_dataset, cnn, criterion, optimizer, batch_size, device, global_pooling_type):
    train_n_batches = len(train_dataset)
    train_n_instances = train_n_batches * batch_size
    val_n_batches = len(val_dataset)
    val_n_instances = val_n_batches * batch_size

    val_results = {'accuracy': [],
                   'loss': [],
                   'epoch': []}
    cnn.train()
    for epoch in range(epochs):
        running_loss = 0.0
        n_correct = 0
        start = time.time()
        for i, dataloader in enumerate(train_dataset.datasets):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = cnn(inputs, global_pooling_type)
                predicted_classes = torch.argmax(outputs, dim=1)
                n_correct += (predicted_classes == labels).sum().item()
                loss = criterion(outputs, labels.type(torch.long))
                running_loss += loss.item()
                loss.backward()
                optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, loss: {(running_loss / train_n_batches):.3f}, accuracy: {(n_correct / train_n_instances):.3f}, time: {(time.time() - start):2f} seconds')

        val_n_correct = 0
        val_running_loss = 0.0
        for i, dataloader in enumerate(val_dataset.datasets):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = cnn(inputs, global_pooling_type)
                predicted_classes = torch.argmax(outputs, dim=1)
                val_n_correct += (predicted_classes == labels).sum().item()
                loss = criterion(outputs, labels.type(torch.long))
                val_running_loss += loss.item()
        print(f"Epoch {epoch + 1}, val accuracy: {(val_n_correct / val_n_instances):.3f}")
        val_results['accuracy'].append(val_n_correct / val_n_instances)
        val_results['loss'].append(val_running_loss)
        val_results['epoch'].append(epoch + 1)

    torch.save(cnn, f'models/cnn_mnist_{datetime.now()}.pt')
    return val_results

def global_pooling_cnn(global_pooling_type, opt, learning_rate=0.001, momentum=None):
    batch_size = get_batch_size()
    N = 81

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    train_batches_total, val_batches_total = [], []
    for resolution in [32, 48, 64]:
        # Use the ImageFolder dataloader to load the images of the specified resolution
        imagefolder_train = torchvision.datasets.ImageFolder(f'MNIST_dataset/MNIST_variable_resolution/mnist-varres-pre-processed/train/{resolution}', transform=transform)

        # Calculate the no. images for the train- and test set ((5/6)th & (1/6)th respectively)
        n_train = int(len(imagefolder_train.samples) * (5 / 6))
        n_validation = len(imagefolder_train.samples) - n_train

        # Split the train and validation set into batches (each batch now contains tensors of the same shape)
        train_batches, val_batches = split_train_validation(
            training_data=imagefolder_train,
            batch_size=batch_size,
            n_train=n_train,
            n_validation=n_validation
        )
        # Concat to the lists
        train_batches_total.append(train_batches)
        val_batches_total.append(val_batches)
    # Concat using the ConcatDataset pytorch function
    train_batches_concat = torch.utils.data.ConcatDataset(train_batches_total)
    val_batches_concat = torch.utils.data.ConcatDataset(val_batches_total)

    return cnn(train=train_batches_concat,
               val=val_batches_concat,
               batch_size=batch_size,
               N=N,
               global_pooling_type=global_pooling_type,
               learning_rate=learning_rate,
               opt=opt,
               momentum=momentum)

if __name__ == '__main__':
    global_pooling_cnn(global_pooling_type='mean', opt='adam')
