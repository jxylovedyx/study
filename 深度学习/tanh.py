from matplotlib import display
import matplotlib.pyplot as plt
import torch    
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.tanh(x)
d2l.plot(x.detach().numpy(), y.detach().numpy(), 'tanh', figsize=(5,2.5))
plt.show()

y.sum().backward()
d2l.plot(x.detach().numpy(), x.grad.numpy(), 'grad of tanh', figsize=(5,2.5))
plt.show()