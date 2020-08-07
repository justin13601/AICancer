import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import *
from torchvision import datasets, models, transforms
from torchsummary import summary
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable


def get_model_name(name, batch_size, learning_rate, epoch):
    """ 
    Returns a path name for the model consisting of all the hyperparameter values.

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size
                                                   ,
                                                   learning_rate,
                                                   epoch)
    return path


def normalize_label(labels):
    """
    Returns a normalized label given a tensor containing 2 possible values.

    Args:
        labels: a 1D tensor containing two possible scalar values
    Returns:
        tensor: A tensor normalize to 0/1 value
    """
    max_val = torch.max(labels)
    min_val = torch.min(labels)
    norm_labels = (labels - min_val)//(max_val - min_val)
    return norm_labels


def evaluate(net, loader, criterion, use_cuda=False):
    """ 
    Evaluates the network on a dataset.

    Args:
        net: PyTorch neural network object
        loader: PyTorch data loader for the validation set
        criterion: The loss function
    Returns:
        err: A scalar for the avg classification error over the validation set
        loss: A scalar for the average loss function over the validation set
     """
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        # Convert labels to 0/1
        labels = normalize_label(labels) 
        if use_cuda and torch.cuda.is_available():
            inputs= inputs.cuda()
            labels = labels.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels.float())
        corr = (outputs > 0.0).squeeze().long() != labels
        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)
    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)
    return err, loss


def plot_training_curve(path):
    """ 
    Plots the training curve for a model run, given the .csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Error")
    n = len(train_err)
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.legend(loc='best')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
