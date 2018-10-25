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
        self.conv1 = ???
        self.conv2 = ???
        self.conv3 = ???
        self.fc1 = ???
        self.fc2 = ???
        self.fc3 = ???
        #

        self.init_weights()
    
    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5*5*C_in))
            nn.init.constant_(conv.bias, 0.0)
        
        # TODO: initialize the parameters for [self.fc1, self.fc2, self.fc3]

        #
        
    def forward(self, x):
        N, C, H, W = x.shape
        
        # TODO: forward pass

        #

        return z
    