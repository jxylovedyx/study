import torch
from torch import nn
import torch.nn.functional as F

class Centerlayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(in_units,units))
        self.bias = nn.Parameter(torch.zeros(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

net = Centerlayer()
net(X)

clone = Centerlayer()
clone.load_state_dict(net.state_dict())
clone.eval