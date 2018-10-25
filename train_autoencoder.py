'''
EECS 445 - Introduction to Machine Learning
Fall 2018 - Project 2
Train Autoencoder
    Trains an autoencoder to learn a sparse representation of image data
    Periodically outputs training information, and saves model checkpoints
    Usage: python train_autoencoder.py
'''
import torch
import numpy as np
import utils
import matplotlib.pyplot as plt
from dataset import get_train_val_test_loaders
from model.autoencoder import Autoencoder, NaiveRecon
from train_common import *
from utils import config

def _train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    # TODO: complete the training step
    for i, (X, y) in enumerate(data_loader):
        # clear parameter gradients
        ???
        #

        # forward + backward + optimize
        ???
        #
    #

def _evaluate_epoch(axes, tr_loader, val_loader, model, criterion, epoch, stats):
    """
    Evaluates the `model` on the train and validation set.
    """
    with torch.no_grad():
        running_loss = []
        for X, y in tr_loader:
            _, recon = model(X)
            running_loss.append(criterion(recon, X))
        tr_loss = np.mean(running_loss)
    with torch.no_grad():
        running_loss = []
        for X, y in val_loader:
            _, recon = model(X)
            running_loss.append(criterion(recon, X))
        val_loss = np.mean(running_loss)
    stats.append([val_loss, tr_loss])
    utils.log_ae_training(epoch, stats)
    utils.update_ae_training_plot(axes, epoch, stats)

def main():
    # data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        num_classes=config('autoencoder.num_classes'))

    # Model
    model = Autoencoder(config('autoencoder.ae_repr_dim'))

    # TODO: define loss function, and optimizer
    criterion = ???
    optimizer = ???
    #

    # Attempts to restore the latest checkpoint if exists
    print('Loading autoencoder...')
    model, start_epoch, stats = restore_checkpoint(model,
        config('autoencoder.checkpoint'))

    axes = utils.make_ae_training_plot()

    # Evaluate the randomly initialized model
    _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, start_epoch,
        stats)

    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config('autoencoder.num_epochs')):
        # Train model
        _train_epoch(tr_loader, model, criterion, optimizer)
        _train_epoch(te_loader, model, criterion, optimizer)

        # Evaluate model
        _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, epoch+1,
            stats)

        # Save model parameters
        save_checkpoint(model, epoch+1, config('autoencoder.checkpoint'), stats)

    print('Finished Training')

    # Save figure and keep plot open
    utils.save_ae_training_plot()
    utils.hold_training_plot()

if __name__ == '__main__':
    main()