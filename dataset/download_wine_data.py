#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载 Wine Quality 数据集
"""

import os
import pandas as pd
import urllib.request

# 数据集 URL（UCI Machine Learning Repository）
RED_WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

# 保存路径
dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
os.makedirs(dataset_dir, exist_ok=True)

red_wine_path = os.path.join(dataset_dir, 'winequality-red.csv')
white_wine_path = os.path.join(dataset_dir, 'winequality-white.csv')
combined_path = os.path.join(dataset_dir, 'winequality.csv')

print("正在下载红酒质量数据集...")
urllib.request.urlretrieve(RED_WINE_URL, red_wine_path)
print(f"红酒数据已保存到：{red_wine_path}")

print("正在下载白酒质量数据集...")
urllib.request.urlretrieve(WHITE_WINE_URL, white_wine_path)
print(f"白酒数据已保存到：{white_wine_path}")

# 读取并合并数据
print("合并红酒和白酒数据...")
red_wine = pd.read_csv(red_wine_path, sep=';')
white_wine = pd.read_csv(white_wine_path, sep=';')

# 添加酒类标签
red_wine['wine_type'] = 0  # 红酒
white_wine['wine_type'] = 1  # 白酒

# 合并数据集
combined = pd.concat([red_wine, white_wine], ignore_index=True)

# 保存合并后的数据集
combined.to_csv(combined_path, index=False)
print(f"合并后的数据已保存到：{combined_path}")

print(f"\n数据集统计信息:")
print(f"总样本数：{len(combined)}")
print(f"特征数量：{len(combined.columns) - 1}")  # 不包括 target
print(f"特征名称：{list(combined.columns[:-1])}")
print(f"\n质量等级分布:")
print(combined['quality'].value_counts().sort_index())
print(f"\n酒类分布:")
print(f"红酒：{len(red_wine)} 个样本")
print(f"白酒：{len(white_wine)} 个样本")

print("\n数据集下载完成!")
