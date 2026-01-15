import torch
from d2l import torch as d2l
from torch.utils import data
from torch import nn

batch_size = 2
input_featyres = 4
output_features = 3

X = torch.tensor([
    [1.0, 2, 4, 8]
    [0.5, 1, 0.7,0.9]])
torch.random.manual_seed(42) # 固定随机种子方便复现
W = torch.randn(d, q)