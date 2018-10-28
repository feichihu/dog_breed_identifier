'''
EECS 445 - Introduction to Machine Learning
Fall 2018 - Project 2
CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.cnn import CNN
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: define each layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5), stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5,5), stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5,5), stride=2, padding=2)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)
        self.init_weights()
    
    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5*5*C_in))
            nn.init.constant_(conv.bias, 0.0)
        for fc in [self.fc1, self.fc2, self.fc3]:
            C_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1 / sqrt(C_in))
            nn.init.constant_(fc.bias, 0.0)
        # TODO: initialize the parameters for [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        N, C, H, W = x.shape
        X = x
        z = []
            #print(X)
            #print(X.shape)
        X = self.conv1(X)
        X = F.relu(X)
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = X.view(-1, 32*4*4)
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        X = self.fc3(X)
        return X
    