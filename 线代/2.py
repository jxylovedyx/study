import torch

# 1. 准备数据
X = torch.tensor([[1, 2], [3, 4]])
Y = torch.tensor([[2, 2], [2, 2]])

# -------------------------------------------------
# 场景一：普通写法 (Z = X + Y) -> 地址会变！
# -------------------------------------------------
print("--- 场景一：普通赋值 (Z = ...) ---")
Z = torch.zeros_like(Y)  # 创建一个全0矩阵，形状和Y一样
before_id = id(Z)
print(f"Z 修改前的地址: {before_id}")

# 执行普通加法
Z = X + Y

after_id = id(Z)
print(f"Z 修改后的地址: {after_id}")
print(f"地址一样吗？ {before_id == after_id}")  # 结果应该是 False


# -------------------------------------------------
# 场景二：原地操作写法 (Z[:] = ...) -> 地址不变！
# -------------------------------------------------
print("\n--- 场景二：切片赋值 (Z[:] = ...) ---")
Z = torch.zeros_like(Y) # 重置 Z
before_id = id(Z)
print(f"Z 修改前的地址: {before_id}")

# 执行原地操作！
# [:] 的意思是："不要换掉 Z 这个容器，而是把数据填进 Z 这个容器里"
Z[:] = X + Y

after_id = id(Z)
print(f"Z 修改后的地址: {after_id}")
print(f"地址一样吗？ {before_id == after_id}")  # 结果应该是 True


# -------------------------------------------------
# 场景三：自加写法 (X += Y) -> 地址不变！
# -------------------------------------------------
print("\n--- 场景三：自加操作 (X += Y) ---")
before_id = id(X)
print(f"X 修改前的地址: {before_id}")

# 执行自加
X += Y

after_id = id(X)
print(f"X 修改后的地址: {after_id}")
print(f"地址一样吗？ {before_id == after_id}")  # 结果应该是 True