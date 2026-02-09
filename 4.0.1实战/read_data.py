import numpy as np
import pandas as pd
import torch
from torch import nn
import data

data.DATA_HUB['kaggle_house'] = (data.DATA_URL + 'kaggle_house_pred_train.csv',
                            '585e9cc93e70b39160e7921475f9bcd47d62293')

data.DATA_HUB['kaggle_house_test'] = (data.DATA_URL + 'kaggle_house_pred_test.csv',
                                 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(data.download('kaggle_house'))
test_data = pd.read_csv(data.download('kaggle_house_test'))
print(train_data.shape)
print(test_data.shape)

# 1. 合并训练集和测试集的特征（不包含标签）
# train_data.iloc[:, 1:-1] : 去掉 Id（第 0 列）和 SalePrice（最后一列）
# test_data.iloc[:, 1:]    : 去掉 Id（第 0 列）
# 目的：统一特征工程规则，保证 train / test 的特征维度完全一致
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 2. 找出数值型特征（非 object 类型）
# object 类型一般是字符串（类别特征）
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

# 3. 对数值特征做标准化（Z-score）
# x -> (x - mean) / std
# 目的：
#   - 消除量纲差异
#   - 加快梯度下降收敛
#   - 让不同特征在同一尺度上
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)

# 4. 处理数值特征中的缺失值
# 在标准化之后，均值约为 0
# 用 0 填充等价于：该样本在该特征上处于“平均水平”
# 是对神经网络非常友好的处理方式
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 5. 对类别特征进行 one-hot 编码
#   - 每个类别值变成一个 0/1 特征
#   - dummy_na=True：把缺失值也当作一个独立类别
# 目的：
#   - 避免给类别强加“大小关系”
#   - 让模型能够利用“是否缺失”这一信息
all_features = pd.get_dummies(all_features, dummy_na=True)

# 查看 one-hot 后的特征维度（通常会大幅增加）
all_features.shape

# 6. 根据训练集样本数，把合并后的特征再拆分回 train / test
n_train = train_data.shape[0]

# 前 n_train 行对应训练集特征
train_features = torch.tensor(
    all_features[:n_train].values,
    dtype=torch.float32
)

# 后面的行对应测试集特征
test_features = torch.tensor(
    all_features[n_train:].values,
    dtype=torch.float32
)

# 7. 构造训练集标签（回归目标）
# SalePrice 是连续值，因此用 float32
# view(-1, 1) 把标签变成 (N, 1)，与网络输出形状一致
train_labels = torch.tensor(
    train_data.SalePrice.values,
    dtype=torch.float32
).view(-1, 1)

# （下面这几行是重复代码，功能与上面完全一致，可以删掉）
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values, dtype=torch.float32
).view(-1, 1)




