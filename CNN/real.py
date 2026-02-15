import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                 stride=1, padding=1, bias=True)

x = torch.randn(8, 3, 64, 64)   # (N,C,H,W)
y = conv(x)
print(y.shape)                 # (8,16,64,64)
