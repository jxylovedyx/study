import torch
from torch import nn
from d2l import torch as d2l

conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
X = X.reshape((1,1,6,8))
Y = Y.reshape((1,1,6,7))
lr = 3e-2

for epoch in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    l.sum().backward()
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.weight.grad.zero_()
    print(f'epoch {epoch + 1}, loss {l.sum().item()}')

def comp_conv2d(conv2d, X):
    X = X.reshape((1,1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])