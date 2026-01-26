import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l
import numpy as np

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
d2l.plot(torch.arange(-8.0, 8.0, 0.1).numpy(), y.detach().numpy(), 'sigmoid', figsize=(5,2.5))
plt.show()

y.sum().backward()
d2l.plot(torch.arange(-8.0, 8.0, 0.1).numpy(), x.grad.numpy(), 'grad of sigmoid', figsize=(5,2.5))
plt.show()