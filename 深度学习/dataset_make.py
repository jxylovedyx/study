import numpy as np
import math
import torch

# ===== 1) 手动造 3 个样本的 x，方便看清楚 =====
features = np.array([[2.0], [-1.0], [0.5]])   # shape (3, 1)
max_degree = 4  # 只做 0~3 次（4维），更好理解

# 真实权重：对应 [x^0, x^1, x^2, x^3]
true_w = np.array([5.0, 1.2, -3.4, 5.6])      # shape (4,)

print("features (x):")
print(features)
print("features shape:", features.shape)
print()

# ===== 2) 把 x 扩展成多项式特征矩阵： [x^0, x^1, x^2, x^3] =====
exponents = np.arange(max_degree).reshape(1, -1)  # shape (1, 4) => [[0,1,2,3]]
poly_features = np.power(features, exponents)      # broadcast => shape (3, 4)

print("exponents:")
print(exponents)
print("poly_features = x^i (未缩放):")
print(poly_features)
print("poly_features shape:", poly_features.shape)
print()

# ===== 3) 用 gamma(i+1)=i! 做缩放： x^i / i! =====
poly_scaled = poly_features.copy()
for i in range(max_degree):
    poly_scaled[:, i] /= math.gamma(i + 1)  # i!

print("factorials (i!):", [math.gamma(i + 1) for i in range(max_degree)])
print("poly_features (缩放后 x^i / i!):")
print(poly_scaled)
print()

# ===== 4) 计算 labels：y = poly_scaled @ true_w =====
labels = poly_scaled @ true_w  # shape (3,)

print("true_w:")
print(true_w)
print("labels (未加噪声):")
print(labels)
print()

# ===== 5) 加一点噪声（可选） =====
np.random.seed(0)
noise = np.random.normal(scale=0.1, size=labels.shape)
labels_noisy = labels + noise

print("noise:")
print(noise)
print("labels (加噪声后):")
print(labels_noisy)
print()

# ===== 6) 转成 torch tensor（和你书里一致） =====
true_w_t = torch.tensor(true_w, dtype=torch.float32)
features_t = torch.tensor(features, dtype=torch.float32)
poly_scaled_t = torch.tensor(poly_scaled, dtype=torch.float32)
labels_noisy_t = torch.tensor(labels_noisy, dtype=torch.float32)

print("Torch tensors dtypes:")
print("true_w_t:", true_w_t.dtype, "shape:", tuple(true_w_t.shape))
print("features_t:", features_t.dtype, "shape:", tuple(features_t.shape))
print("poly_scaled_t:", poly_scaled_t.dtype, "shape:", tuple(poly_scaled_t.shape))
print("labels_noisy_t:", labels_noisy_t.dtype, "shape:", tuple(labels_noisy_t.shape))
