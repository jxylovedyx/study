import torch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(X)
print(a+X)
print(X.sum())