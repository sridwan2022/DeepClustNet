import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, dims, act='relu'):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        for i in range(len(dims)-1):
            self.encoder.add_module('encoder_%d' % i, nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                self.encoder.add_module('encoder_act_%d' % i, nn.ReLU())
        for i in range(len(dims)-1, 0, -1):
            self.decoder.add_module('decoder_%d' % i, nn.Linear(dims[i], dims[i-1]))
            if i > 1:
                self.decoder.add_module('decoder_act_%d' % i, nn.ReLU())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def generator(dataloader, x, y=None, sample_weight=None):
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.view(inputs.size(0), -1)
        if y is None:
            yield inputs, inputs
        else:
            yield inputs, y[i]
        if sample_weight is not None:
            yield inputs, y[i], sample_weight[i]

def random_transform(x, datagen):
    dataloader = DataLoader(datagen, batch_size=len(x), shuffle=False, num_workers=2)
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.view(inputs.size(0), -1)
        return inputs.numpy()
