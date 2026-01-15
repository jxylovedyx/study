import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
a = torch.tensor([3.0, 4.0])

# 1. 计算 y = x · a (点积)
# y = 1*3 + 2*4 = 11
y = torch.dot(x, a)
print(y)
# 2. 求导
y.backward()
print(x.grad)
