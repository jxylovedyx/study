import torch
from torch import nn
from d2l import torch as d2l

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for i, module in enumerate(args):
            self.add_module(str(i), module)

    def forward(self, X):
        for module in self._modules.values():
            X = module(X)
        return X