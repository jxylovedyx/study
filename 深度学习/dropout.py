import torch
from torch import nn
from d2l import torch as d2l

def dropout_layer(X, drop_prob):
    assert 0 <= drop_prob <= 1
    if drop_prob == 1:
        return torch.zeros_like(X)
    if drop_prob == 0:
        return X
    mask = (torch.rand(X.shape) > drop_prob).float()
    return mask * X / (1.0 - drop_prob)

X = torch.arange(16).view(2, 8)
