import torch
import matplotlib.pyplot as plt
x=torch.arange(4.0,requires_grad=True)
print(x)
y=2*torch.dot(x,x)
print(x.grad)
print(y)