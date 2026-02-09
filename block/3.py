import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = F.relu(self.linear(X))
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        return X.sum()