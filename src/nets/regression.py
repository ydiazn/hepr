import torch
import torch.nn.functional as F


class RegressionNet(torch.nn.Module):
    '''
    NN model for regression
    '''
    def __init__(self, n_feature, n_output, n_hidden=10):
        super().__init__()
        # hidden layer
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        # output layer
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # activation function for hidden layer
        x = F.relu(self.hidden(x))
        # linear output
        x = self.predict(x)
        return x


class RegressionNet2(torch.nn.Module):
    def __init__(self, n_feature, n_output, n_hidden=10):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        p = F.relu(self.hidden(x))
        p = self.predict(p)
        q = F.relu(self.hidden(x))
        q = self.predict(q)

        return p, q