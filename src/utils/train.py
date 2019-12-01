import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

from src.nets.regression import RegressionNet


def regresion_mse(data, target, output, net, epochs=100):

    # torch can only train on Variable, so convert them to Variable
    data = torch.from_numpy(data).float()
    target = torch.from_numpy(target).float()
    x, y = Variable(data), Variable(target)

    # define the network
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()

    # train the network
    for epoch in range(epochs):
        # input x and predict based on x
        prediction = net(x)
        # must be (1. nn output, 2. target)
        loss = loss_func(prediction, y)
        # clear gradients for next train
        optimizer.zero_grad()
        # backpropagation, compute gradients
        loss.backward()
        # apply gradients
        optimizer.step()
        print('Step = %d, Loss = %.4f' % (epoch, loss.data.numpy()))
    
    #assert False, output
    torch.save(net, output)
