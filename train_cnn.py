"""
EECS 445 - Introduction to Machine Learning
Fall 2018 - Project 2
Train CNN
    Trains a convolutional neural network to classify images
    Periodically outputs training information, and saves model checkpoints
    Usage: python train_cnn.py
"""
import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
import dataset
from model.cnn import CNN
from train_common import *
from utils import config
import utils

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def _train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    for i, (X, y) in enumerate(data_loader):
        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

def _evaluate_epoch(axes, tr_loader, val_loader, model, criterion, epoch, stats):
    """
    Evaluates the `model` on the train and validation set.
    """
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in tr_loader:
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
        train_loss = np.mean(running_loss)
        train_acc = correct / total
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in val_loader:
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
        print("loss",running_loss)
        val_loss = np.mean(running_loss)
        val_acc = correct / total
    stats.append([val_acc, val_loss, train_acc, train_loss])
    utils.log_cnn_training(epoch, stats)
    utils.update_cnn_training_plot(axes, epoch, stats)

def get_data_by_label(dataset):
    data = []
    for i, (X, y) in enumerate(dataset):
        for c in range(config('autoencoder.num_classes')):
            batch = X[(y == c).nonzero().squeeze(-1)]
            if len(data) <= c:
                data.append(batch)
            else:
                data[c] = torch.cat((data[c], batch))
    return data

def main():
    # Data loaders
    tr_loader, va_loader, te_loader, get_semantic_label = get_train_val_test_loaders(
        num_classes=config('cnn.num_classes'))

    # Model
    model = CNN()
    
    # TODO: define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #

    print('Number of float-valued parameters:', count_parameters(model))

    # Attempts to restore the latest checkpoint if exists
    print('Loading cnn...')
    model, start_epoch, stats = restore_checkpoint(model, config('cnn.checkpoint'))

    axes = utils.make_cnn_training_plot()

    # Evaluate the randomly initialized model
    _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, start_epoch,
        stats)
    #reprot each class
    dataset = get_data_by_label(va_loader)
    len_ = 0
    y_true = []
    y_pred = []
    accuracy = []
    total = 0
    correct = 0
    for c in range(5):
        X = dataset[c]
        output = model(X)
        predicted = predictions(output.data)
        y = [c]*X.shape[0]
        y = torch.LongTensor(y)
        y_true.append(y)
        y_pred.append(predicted)
        total += len(y)
        correct += (predicted == y).sum().item()
        accuracy.append(correct/total)
    for c, p in enumerate(accuracy):
        print('Class {}: {} '
            .format(get_semantic_label(c), p)) 
    '''finish'''
    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config('cnn.num_epochs')):
        # Train model
        _train_epoch(tr_loader, model, criterion, optimizer)
        
        # Evaluate model
        _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, epoch+1,
            stats)

        # Save model parameters
        save_checkpoint(model, epoch+1, config('cnn.checkpoint'), stats)

    print('Finished Training')

    # Save figure and keep plot open
    utils.save_cnn_training_plot()
    utils.hold_training_plot()

if __name__ == '__main__':
    main()
