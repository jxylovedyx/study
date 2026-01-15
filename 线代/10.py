import torch
x = torch.arange(4, dtype = torch.float32)
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)