import torch
import torchvision
import time
import torch.nn as nn
import torch.optim as optim
import os
from question7 import split_train_validation
from torchvision.transforms import ToTensor

class CNN_MNIST(nn.Module):
    def __init__(self, batch_size, device):
        super().__init__()
        self.batch_size = batch_size

        self.relu = nn.ReLU()
        self.conv1_2d = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1, device=device)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2_2d = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, device=device)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3_2d = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, device=device)
        self.max_pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.linear1 = nn.Linear(in_features=64 * 3 * 3, out_features=10, device=device)
        self.linear2 = nn.Linear(10, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1_2d(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.conv2_2d(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        x = self.conv3_2d(x)
        x = self.relu(x)
        x = self.max_pool3(x)
        x = torch.reshape(input=x, shape=(self.batch_size, 64 * 3 * 3))
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

def training_loop(epochs, dataset, cnn, criterion, optimizer):
    n_batches = len(dataset)
    n_instances = n_batches * batch_size
    cnn.train()
    for epoch in range(epochs):
        running_loss = 0.0
        n_correct = 0
        start = time.time()
        for inputs, labels in dataset:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = cnn(inputs)
            predicted_classes = torch.argmax(outputs, dim=1)
            n_correct += (predicted_classes == labels).sum().item()
            loss = criterion(outputs, labels.type(torch.long))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, loss: {(running_loss / n_batches):.3f}, accuracy: {(n_correct / n_instances):.3f}, time: {(time.time() - start):2f} seconds')

if __name__ == '__main__':
    batch_size = 16
    epochs = 20
    learning_rate = 0.001
    device_name = 'cpu'

    if torch.cuda.is_available():
        device_name = 'cuda:0'
    elif torch.backends.mps.is_available():
        device_name = 'mps'
    device = torch.device(device_name)

    path_dataset = 'MNIST_dataset'
    if not os.path.exists(path_dataset):
        os.mkdir(path_dataset)
    train = torchvision.datasets.MNIST(root=path_dataset, train=True, download=True, transform=ToTensor())
    test = torchvision.datasets.MNIST(root=path_dataset, train=False, download=True, transform=ToTensor())
    train_batches, val_batches = split_train_validation(training_data=train, batch_size=batch_size)

    cnn = CNN_MNIST(batch_size=batch_size, device=device)
    cnn.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    training_loop(epochs=epochs, dataset=train_batches, cnn=cnn, criterion=criterion, optimizer=optimizer)
