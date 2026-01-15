import torch

# 创建一个长度为 10亿 的向量
# 这大概会占用 4GB - 8GB 的内存
try:
    big_vector = torch.arange(9000000000)
    print("创建成功！长度是:", len(big_vector))
except RuntimeError as e:
    print("内存炸了！错误信息:", e)