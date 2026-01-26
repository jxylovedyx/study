import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l

x= torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)

d2l.plot(x.detach().numpy(), y.detach().numpy(), 'relu', figsize=(5,2.5))
plt.show()

y.backward(torch.ones_like(x))
d2l.plot(x.detach().numpy(), x.grad.numpy(), 'grad of relu', figsize=(5,2.5))
plt.show()