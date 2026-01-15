import torch
A = torch.arange(24).reshape(2, 3, 4)
sum_A = A.sum(dim=1, keepdims=True)
print(sum_A)
print(A.cumsum(dim=0))