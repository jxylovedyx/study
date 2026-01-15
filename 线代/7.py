import torch
A = torch.randint(0,20,(5,4))
B = torch.arange(20,dtype=torch.float).reshape(5,4)
print(A)
print(A+B)
print(A*B)