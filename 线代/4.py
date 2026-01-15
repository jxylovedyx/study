import torch
import pandas as pd
import numpy as np

data = pd.read_csv(r'C:/Users/余鸿寒潇/PycharmProjects/pytorch_study/data/sensor_data.csv')
nan_counts = data.isnull().sum()
print("每列缺失值数量:\n", nan_counts)
target_col = nan_counts.idxmax()
print("\n缺失最多的列是:", target_col)
data.drop(columns=target_col, inplace=True)
data = data.fillna(0)
final_tensor = torch.tensor(data.to_numpy(dtype=float))
print("\n最终的张量:\n", final_tensor)
