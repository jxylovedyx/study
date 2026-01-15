import torch
A = torch.arange(24,dtype=float).reshape(1,2, 3, 4)
print(A)
print(A.sum(axis=0,keepdim=True))
print(A.sum(axis=1,keepdim=True))
print(A.sum(axis=2,keepdim=True))
print(A.sum(axis=3,keepdim=True))
print(torch.linalg.norm(A))