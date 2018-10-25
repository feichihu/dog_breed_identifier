'''
EECS 445 - Introduction to Machine Learning
Fall 2018 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Challenge(nn.Module):
    def __init__(self):
        super().__init__()
        
        # TODO:

        #

        self.init_weights()

    def init_weights(self):
        # TODO:

        #
        
    def forward(self, x):
        N, C, H, W = x.shape

        # TODO:

        #

        return z
