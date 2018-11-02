"""
EECS 445 - Introduction to Machine Learning
Fall 2018 - Project 2
Train Autoencoder
    Trains an autoencoder to learn a sparse representation of images data
    Periodically outputs training information, and saves model checkpoints
    Usage: python train_autoencoder.py
"""
import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
import dataset
from model.autoencoder import Autoencoder, AutoencoderClassifier
from train_common import *
from utils import config
import utils

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

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
    # data loaders
    tr_loader, va_loader, te_loader, get_semantic_label = get_train_val_test_loaders(
        num_classes=config('autoencoder.classifier.num_classes'))

    ae_classifier = AutoencoderClassifier(config('autoencoder.ae_repr_dim'),
        config('autoencoder.classifier.num_classes'))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ae_classifier.parameters(),
        lr=config('autoencoder.classifier.learning_rate'))

    # freeze the weights of the encoder
    for name, param in ae_classifier.named_parameters():
        if 'fc1.' in name or 'fc2.' in name:
            param.requires_grad = False

    # Attempts to restore the latest checkpoint if exists
    print('Loading autoencoder...')
    ae_classifier, _, _ = restore_checkpoint(ae_classifier,
        config('autoencoder.checkpoint'), force=True, pretrain=True)
    print('Loading autoencoder classifier...')
    ae_classifier, start_epoch, stats = restore_checkpoint(ae_classifier,
        config('autoencoder.classifier.checkpoint'))

    axes = utils.make_cnn_training_plot()

    # Evaluate the randomly initialized model
    _evaluate_epoch(axes, tr_loader, va_loader, ae_classifier, criterion,
        start_epoch, stats)
    
    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config('autoencoder.classifier.num_epochs')):
        # Train model
        _train_epoch(tr_loader, ae_classifier, criterion, optimizer)
        
        # Evaluate model
        _evaluate_epoch(axes, tr_loader, va_loader, ae_classifier, criterion,
            epoch+1, stats)

        # Save model parameters
        save_checkpoint(ae_classifier, epoch+1,
            config('autoencoder.classifier.checkpoint'), stats)

    print('Finished Training')
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
        output = ae_classifier(X)
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
    # Keep plot open
    utils.save_cnn_training_plot()
    utils.hold_training_plot()

if __name__ == '__main__':
    main()