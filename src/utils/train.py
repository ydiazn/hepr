from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import datasets, transforms, models
import imageio
from PIL import Image

from src.nets.regression import RegressionNet


def train(model, image_dir, targets, optimizer, epoch, loss_func, preprocess, log_interval=2):
    model.train()
    for i, file in enumerate(sorted(Path(image_dir).iterdir())):
        # Setp data and target
        image = Image.open(file)
        input = preprocess(image)
        input = input.unsqueeze(0)
        target = targets[i]
        target = torch.from_numpy(target).float()
        p, q = target
        # optimization
        optimizer.zero_grad()
        ps, qs = model(input)[0]
        ps = ps.unsqueeze(0)
        qs = qs.unsqueeze(0)

        loss = loss_func(ps, p)
        loss += loss_func(qs, q)
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
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    # define the network
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_func = torch.nn.MSELoss()

    #assert False, output
    for epoch in range(1, epochs + 1):
        train(model, image_dir, targets, optimizer, epoch, loss_func, preprocess)

    torch.save(model, output)


def linear_network_block(coeficients, targets, output_file, epochs=100):

    # Setup
    lr = 0.001
    momentum = 0.9

    model = RegressionNet(n_feature=64, n_output=2)
    model.train()

    # define the network
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_func = torch.nn.MSELoss()

    #assert False, output
    for epoch in range(1, epochs + 1):
        target = Variable(torch.from_numpy(targets).float())
        data = Variable(torch.from_numpy(coeficients).float())

        # optimization
        optimizer.zero_grad()
        output = model(data)

        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} Loss: {:.6f}'.format(epoch, loss.item()))

    torch.save(model, output_file)

