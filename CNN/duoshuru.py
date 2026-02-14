import torch
from torch import nn
from d2l import torch as d2l

def corr2d_multi_in(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))

def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

def corr2d_multi_in_out_1x1(X, K):
    c_i, c_o = K.shape[1], K.shape[0]
    X = X.reshape((X.shape[0], -1))
    K = K.reshape((c_o, c_i))
    return torch.mm(X, K.T)

