import math
import torch
import time
import numpy as np
from d2l import torch as d2l
import sys
from mytimer import Timer


n=100000
a = torch.ones(n)
b = torch.ones(n)
c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'普通方法用时: {timer.stop():.5f} 秒')

timer.start()
d=a+b
print(f'向量化方法用时: {timer.stop():.5f} 秒')