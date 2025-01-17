import torch
from torchvision import datasets, transforms
import os
import numpy as np

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    x_train = trainset.data.numpy()
    y_train = trainset.targets.numpy()
    x_test = testset.data.numpy()
    y_test = testset.targets.numpy()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape([x.shape[0], -1]) / 255.0
    print('MNIST:', x.shape)
    return x, y

def load_mnist_test():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    _, testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    x = testset.data.numpy()
    y = testset.targets.numpy()
    x = x.reshape([x.shape[0], -1]) / 255.0
    print('MNIST-TEST:', x.shape)
    return x, y

def load_fashion_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    x_train = trainset.data.numpy()
    y_train = trainset.targets.numpy()
    x_test = testset.data.numpy()
    y_test = testset.targets.numpy()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape([x.shape[0], -1]) / 255.0
    print('Fashion MNIST:', x.shape)
    return x, y

def load_usps(data_path='./data/usps'):
    # USPS dataset is not available in torchvision, so we keep this part unchanged
    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64') / 2.
    y = np.concatenate((labels_train, labels_test))
    x = x.reshape([-1, 16*16])
    print('USPS samples', x.shape)
    return x, y

def load_data(dataset):
    dataset = dataset.lower()
    if dataset == 'mnist':
        return load_mnist()
    elif dataset == 'mnist-test':
        return load_mnist_test()
    elif dataset == 'fmnist':
        return load_fashion_mnist()
    elif dataset == 'usps':
        return load_usps()
