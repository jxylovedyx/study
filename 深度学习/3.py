import torch
import torch.nn as nn

layer = nn.Linear(4, 4)
W = layer.weight.data

print("mean:", W.mean().item())
print("std:", W.std().item())
print(W)