import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=5)
        # nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1_batchnorm = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_batchnorm = torch.nn.BatchNorm2d(20)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(500, 32)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc1_batchnorm = torch.nn.BatchNorm1d(32)
        self.fc2 = torch.nn.Linear(32, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1_batchnorm(F.relu(self.pooling(self.conv1(x))))
        x = self.conv2_batchnorm(F.relu(self.pooling(self.conv2(x))))
        x = x.view(batch_size, -1)  # flatten
        x = self.fc1(x)
        x = self.fc1_batchnorm(F.relu(x))
        x = self.fc2(x)

        return x


