"""
neural network models
MNIST dataset, CNN

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=512)  # Adjust input size after pooling dynamically
        self.fc2 = nn.Linear(in_features=512, out_features=10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)

        return x


class LSTMHar(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(LSTMHar, self).__init__()

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_sizes[0], num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_sizes[0], hidden_size=hidden_sizes[1], num_layers=1, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=hidden_sizes[1], out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)

        return x

