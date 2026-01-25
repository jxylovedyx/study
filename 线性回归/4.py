import torch  # 导入 PyTorch（张量运算、自动求导等都在里面）

# -------------------------
# 1) 参数初始化：W, b
# -------------------------

d = 2  # 特征维度：每个样本有2个输入特征（x1, x2）

# 初始化权重 W：形状 (d, 1) = (2, 1)
# 用均值0、标准差0.01的正态分布随机初始化（很小的随机数，训练更稳定）
# requires_grad=True 表示要对它求梯度（训练时要更新它）
W = torch.normal(mean=0.0, std=0.01, size=(d, 1), requires_grad=True)

# 初始化偏置 b：形状 (1,)
# 全0初始化；requires_grad=True 表示也要训练更新
b = torch.zeros(1, requires_grad=True)

# -------------------------
# 2) 定义模型：线性回归前向计算
# -------------------------

def linreg(X, W, b):
    # X 的形状通常是 (batch_size, d)
    # W 的形状是 (d, 1)
    # X @ W 是矩阵乘法： (batch_size, d) @ (d, 1) -> (batch_size, 1)
    # + b 会广播到每个样本上（给每行都加上偏置）
    return X @ W + b

# -------------------------
# 3) 定义损失函数：平方损失（MSE的一种写法）
# -------------------------

def squared_loss(y_hat, y):
    # y_hat 是预测值，形状通常 (batch_size, 1)
    # y 是真实值，可能是 (batch_size, 1) 或 (batch_size,)
    # reshape(y_hat.shape) 把 y 调整到和 y_hat 同形状，避免广播错误
    # (y_hat - y)^2 是逐元素平方误差
    # /2 是一个常见习惯：导数更好看（不影响最优解）
    return ((y_hat - y.reshape(y_hat.shape)) ** 2) / 2

# -------------------------
# 4) 定义优化器：手写 SGD 更新
# -------------------------

def sgd(params, lr, batch_size):
    # torch.no_grad()：告诉 PyTorch“下面这些操作不要记录到计算图里”
    # 因为我们现在是手动更新参数，不需要对更新步骤再求梯度
    with torch.no_grad():
        for p in params:  # 遍历每个参数（这里就是 W 和 b）
            # p.grad 是该参数的梯度（由 loss.backward() 得到）
            # 用 SGD 更新：p = p - lr * grad / batch_size
            # 除以 batch_size 是因为我们这里的 loss 用 sum()，梯度会随batch变大
            p -= lr * p.grad / batch_size

            # 更新完后必须把梯度清零，否则下一轮梯度会“累加”在旧梯度上
            p.grad.zero_()

# -------------------------
# 5) 构造一份“人造数据”(toy data)
# -------------------------

# 真实权重（用来生成数据的“地面真相”）：形状 (2, 1)
true_W = torch.tensor([[2.0], [-3.0]])

# 真实偏置：形状 (1,)
true_b = torch.tensor([0.5])

N = 1000  # 样本数量

# 随机生成输入 X：均值0方差1，形状 (N, d) = (1000, 2)
X = torch.normal(0, 1, (N, d))

# 生成真实 y：y = X @ true_W + true_b + 噪声
# X @ true_W -> (1000,2)@(2,1)=(1000,1)
# true_b 会广播
# 再加一点小噪声，让数据更像现实
y = X @ true_W + true_b + torch.normal(0, 0.1, (N, 1))

# -------------------------
# 6) 训练超参数
# -------------------------

lr = 0.03          # 学习率（步长）
batch_size = 32    # 小批量大小
num_epochs = 5     # 训练轮数（把整个数据集过几遍）

# -------------------------
# 7) 训练循环
# -------------------------

for epoch in range(num_epochs):  # 每一轮 epoch 遍历一遍数据集

    # randperm(N) 生成 0..N-1 的随机排列，用于打乱数据顺序
    idx = torch.randperm(N)

    # 按同一个随机顺序打乱 X 和 y（保持一一对应）
    X_shuf, y_shuf = X[idx], y[idx]

    # 每次取 batch_size 个样本出来训练
    for i in range(0, N, batch_size):
        # 取出当前小批量的输入
        X_batch = X_shuf[i:i+batch_size]

        # 取出当前小批量的标签
        y_batch = y_shuf[i:i+batch_size]

        # 1) 前向：用当前参数算预测值
        y_hat = linreg(X_batch, W, b)

        # 2) 计算损失
        # squared_loss 返回每个样本的损失（形状 (batch,1)）
        # .sum() 把一个batch的损失加起来变成标量，便于 backward()
        loss = squared_loss(y_hat, y_batch).sum()

        # 3) 反向：计算 W,b 的梯度（结果存在 W.grad 和 b.grad 中）
        loss.backward()

        # 4) 参数更新：手写 SGD
        sgd([W, b], lr, batch_size)

    # -------------------------
    # 8) 每个 epoch 结束后算一下全数据的平均训练损失（看训练效果）
    # -------------------------
    with torch.no_grad():  # 评估时不需要构建计算图
        # 用全数据算预测，然后算 mean loss
        train_loss = squared_loss(linreg(X, W, b), y).mean()

    # 打印当前轮的损失
    print(f"epoch {epoch+1}, loss {train_loss.item():.4f}")

# -------------------------
# 9) 打印学到的参数 vs 真实参数
# -------------------------

# W.detach()：把 W 从计算图里“摘出来”，只拿数值
# W.detach().T：转置一下方便打印（从(2,1)变(1,2)）
print("learned W:", W.detach().T)

# 同理打印 b
print("learned b:", b.detach())
