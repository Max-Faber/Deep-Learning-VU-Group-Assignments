import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import os
import time
from datetime import datetime
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
        x = self.softmax(x)
        return x

def training_loop(epochs, train_dataset, val_dataset, cnn, criterion, optimizer, batch_size, device):
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
        for inputs, labels in train_dataset:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = cnn(inputs)
            predicted_classes = torch.argmax(outputs, dim=1)
            n_correct += (predicted_classes == labels).sum().item()
            loss = criterion(outputs, labels.type(torch.long))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, loss: {(running_loss / train_n_batches):.3f}, accuracy: {(n_correct / train_n_instances):.3f}, time: {(time.time() - start):2f} seconds')

        val_n_correct = 0
        val_running_loss = 0.0
        for inputs, labels in val_dataset:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = cnn(inputs)
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

def get_device():
    device_name = 'cpu'

    if torch.cuda.is_available():
        device_name = 'cuda:0'
    elif torch.backends.mps.is_available():
        device_name = 'mps'
    device = torch.device(device_name)
    print(f"Using device: {device_name}")
    return device

def get_batch_size():
    return 16

def load_dataset(transform, batch_size):
    path_dataset = 'MNIST_dataset'
    if not os.path.exists(path_dataset):
        os.mkdir(path_dataset)
    train_transformed = torchvision.datasets.MNIST(root=path_dataset, train=True, download=True, transform=transform)
    train_tensor = torchvision.datasets.MNIST(root=path_dataset, train=True, download=True, transform=ToTensor())

    train_transformed_batches, val_transformed_batches = split_train_validation(training_data=train_transformed, batch_size=batch_size)
    train_tensor_batches, val_tensor_batches = split_train_validation(training_data=train_tensor, batch_size=batch_size)

    del train_transformed, train_tensor, val_transformed_batches, train_tensor_batches
    return train_transformed_batches, val_tensor_batches

def cnn(train, val, batch_size):
    epochs = 10
    learning_rate = 0.001
    device = get_device()

    cnn = CNN_MNIST(batch_size=batch_size, device=device)
    cnn.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    val_results = training_loop(epochs=epochs, train_dataset=train, val_dataset=val, cnn=cnn, criterion=criterion, optimizer=optimizer, batch_size=batch_size, device=device)
    return val_results

def evaluate(test_set, cnn, criterion, device, batch_size):
    val_n_batches = len(test_set)
    val_n_instances = val_n_batches * batch_size
    test_n_correct = 0
    val_running_loss = 0.0
    for inputs, labels in test_set:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = cnn(inputs)
        predicted_classes = torch.argmax(outputs, dim=1)
        test_n_correct += (predicted_classes == labels).sum().item()
        loss = criterion(outputs, labels.type(torch.long))
        val_running_loss += loss.item()
    accuracy = test_n_correct / val_n_instances
    loss = val_running_loss
    return accuracy, loss

if __name__ == '__main__':
    batch_size = get_batch_size()
    train_transformed_batches, val_tensor_batches = load_dataset(transform=ToTensor(), batch_size=batch_size)
    cnn(train=train_transformed_batches, val=val_tensor_batches, batch_size=batch_size)
