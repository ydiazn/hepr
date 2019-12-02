from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import datasets, transforms, models
import imageio

from src.nets.regression import RegressionNet


def train(model, image_dir, targets, optimizer, epoch, loss_func, log_interval=2):
    model.train()
    for i, file in enumerate(sorted(Path(image_dir).iterdir())):
        # Setp data and target
        image = imageio.imread(file)
        image = image.reshape(*image.shape, -1)
        image = image.transpose(3, 2, 0, 1)
        data = torch.from_numpy(image).float()
        data = Variable(data)
        target = targets[i]
        target = torch.from_numpy(target).float()
        target = Variable(target)
        # optimization
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train Epoch: {} Loss: {:.6f}'.format(
                epoch, loss.item()))


def regresion_mse(image_dir, targets, output):

    # Setup
    lr = 0.001
    momentum = 0.9
    epochs=100
    targets = targets.reshape(-1, 2, 1)

    # Convolutional neural network (ResNet18)
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # define the network
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_func = torch.nn.MSELoss()

    #assert False, output
    for epoch in range(1, epochs + 1):
        train(model, image_dir, targets, optimizer, epoch, loss_func)
