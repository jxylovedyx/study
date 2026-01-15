import torch
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
total_sum = X.sum()
sum_by_column = X.sum(dim=0)
sum_by_row = X.sum(dim=1)
print(total_sum)
print(sum_by_column)
print(sum_by_row)