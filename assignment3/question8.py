import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from question7 import split_train_validation, gen_batches
from torchvision.transforms import ToTensor

class CNN_MNIST(nn.Module):
    def __init__(self, batch_size, device):
        super().__init__()
        self.batch_size = batch_size

        self.relu = nn.ReLU()
        self.conv1_2d = nn.Conv2d(1, 16, 3, 1, 1, device=device)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2_2d = nn.Conv2d(16, 32, 3, 1, 1, device=device)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3_2d = nn.Conv2d(32, 64, 3, 1, 1, device=device)
        self.max_pool3 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(64 * 3 * 3, 10, device=device)
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

def training_loop(epochs, X, Y, cnn, criterion, optimizer):
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(len(X)):
            inputs, labels = X[i], Y[i]

            optimizer.zero_grad()
            outputs = cnn(inputs)
            outputs = torch.max(outputs, dim=1)
            loss = criterion(outputs.values, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
                running_loss = 0.0

if __name__ == '__main__':
    batch_size = 16
    epochs = 5
    learning_rate = 0.001
    device_name = 'cpu'

    if torch.cuda.is_available():
        device_name = 'cuda:0'
    elif torch.backends.mps.is_available():
        device_name = 'mps'
    device = torch.device('mps')

    train = torchvision.datasets.MNIST(root='MNIST_dataset', train=True, download=True, transform=ToTensor())
    test = torchvision.datasets.MNIST(root='MNIST_dataset', train=False, download=True, transform=ToTensor())
    train_batches_x, train_batches_y, val_batches_x, val_batches_y = split_train_validation(training_data=train, batch_size=batch_size, device=device)
    # test_batches_x, test_batches_y = gen_batches(test, batch_size=batch_size)

    cnn = CNN_MNIST(batch_size=batch_size, device=device)
    cnn.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    training_loop(epochs=epochs, X=train_batches_x, Y=train_batches_y, cnn=cnn, criterion=criterion, optimizer=optimizer)
