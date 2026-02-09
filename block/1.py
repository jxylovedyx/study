import torch
from torch import nn
from d2l import torch as d2l

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out( F.relu(self.hidden(X)) )
    
net =MLP()
net(x)