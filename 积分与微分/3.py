import sys
sys.path.append('C:/Users/余鸿寒潇/PycharmProjects/pytorch_study')
import torch
import matplotlib.pyplot as plt
from draw import plot 
x = torch.arange(1.0, 3.0, 0.0001)
def f(x):
    return x**3 - 1/x
def df(x):
    x_calc = x.detach().clone() 
    x_calc.requires_grad_(True)  
    y = f(x_calc)
    y.sum().backward()
    return x_calc.grad
plot(x, [f(x).detach(), df(x).detach()],
     xlabel='x', ylabel='f(x)',
     legend=['f(x)', 'dDerivative'])
plt.show()
