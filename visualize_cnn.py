import torch
import torch.nn.functional as F
import numpy as np
import utils
from dataset import get_train_val_test_loaders
from model.cnn import CNN
from train_common import *
from utils import config
import utils
import copy
import math

from matplotlib import pyplot as plt

def visualize_input(i):
    xi, yi = tr_loader.dataset[i]
    fig = plt.figure()
    plt.imshow(utils.denormalize_image(xi.numpy().transpose(1,2,0)))
    plt.axis('off')
    plt.savefig('CNN_viz0_{}.png'.format(yi), dpi=200, bbox_inches='tight')

def visualize_layer1_activations(i):
    xi, yi = tr_loader.dataset[i]
    xi = xi.view((1,3,32,32))
    zi = F.relu(model.conv1(xi))
    zi = zi.detach().numpy()[0]
    sort_mask = np.argsort(model.conv1.weight.detach().numpy().mean(axis=(1,2,3)))
    zi = zi[sort_mask]
    fig, axes = plt.subplots(4, 4, figsize=(10,10))
    for i, ax in enumerate(axes.ravel()):
        ax.axis('off')
        im = ax.imshow(zi[i], cmap='gray')
    fig.suptitle('Layer 1 activations, y={}'.format(yi))
    fig.savefig('CNN_viz1_{}.png'.format(yi), dpi=200, bbox_inches='tight')

if __name__ == '__main__':
    # Attempts to restore from checkpoint
    print('Loading cnn...')
    model = CNN()
    model, start_epoch = restore_checkpoint(model, config('cnn.checkpoint'), force=True)

    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(num_classes=config('cnn.num_classes'))

    # Miniature poodle, y=1
    i = 0
    visualize_input(i)
    visualize_layer1_activations(i)

    # Samoyed, y=0
    i = 48
    visualize_input(i)
    visualize_layer1_activations(i)

    # Great dane, y=3
    i = 131
    visualize_input(i)
    visualize_layer1_activations(i)
