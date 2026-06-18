# simple_cnn_test

> 简单自主科学实验测试项目 — 红酒质量多分类预测

## 📋 项目简介

本项目以 **UCI Wine Quality（红酒+白酒）数据集** 为实验对象，通过系统性地比较 **RandomForest 基线模型** 与 **XGBoost GPU 加速优化模型** 的性能，探索在类别不平衡的多分类任务中，特征工程、模型升级、加权损失和超参数调优对预测效果的提升效果。

### 核心性能指标

| 指标 | 基线 (RandomForest) | 优化 (XGBoost GPU) | 变化 |
|:----:|:-------------------:|:------------------:|:----:|
| **准确率** | 62.69% | **63.62%** | **+0.93%** ✅ |
| **加权平均 F1** | 0.60 | **0.64** | **+0.04** ✅ |
| **宏平均 F1** | 0.30 | **0.39** | **+0.09** ✅ |

## 🚀 优化方案

### 改进标题

**XGBoost GPU 加速 + 特征工程 + 加权损失 + 超参数搜索**

### 改进描述

利用 NVIDIA H20 GPU 的强大算力，将当前 CPU 上运行的 RandomForestClassifier 替换为支持 GPU 加速的 XGBoost 模型，并通过修复被遗漏的关键特征 `wine_type`、引入有意义的交互特征、对类别不平衡问题施加样本权重、以及结合早停机制与超参数调优，系统性提升红酒质量多分类任务的准确率和泛化能力。

### 具体改进措施

| 改进维度 | 具体措施 | 效果 |
|---------|---------|------|
| **模型升级** | RandomForest → XGBoost (`tree_method=hist`, `device=cuda`) | 利用 GPU 加速训练 |
| **特征修复** | 增加被遗漏的 `wine_type` 特征（红酒/白酒类型） | 基础特征从 11→12 维 |
| **特征工程** | 对数变换(4) + 酒类交互(7) + 比例组合(9) + 平方项(4) | 特征扩展至 36 维 |
| **加权损失** | 基于类别逆频率计算样本权重 | 稀有类别召回率提升 |
| **超参数搜索** | RandomizedSearchCV (20次采样, 3折 CV) | 搜索 9 个超参数维度 |
| **早停机制** | early_stopping_rounds=50 | 防止过拟合 |

## 📊 数据集

| 属性 | 值 |
|------|-----|
| 数据集名称 | UCI Wine Quality（红酒 + 白酒质量） |
| 样本总数 | 6,497 |
| 特征数量 | 11 个理化特征 → 优化后扩展为 36 维增强特征 |
| 类别数量 | 7 (质量等级 3-9) |
| 训练集大小 | 5,197 (80%) |
| 测试集大小 | 1,300 (20%) |

## ⚙️ 使用方法

### 环境要求

```bash
git clone <repository_url>
cd simple_cnn_test
```

**Python 依赖：** 请确保已安装以下关键库：

- Python >= 3.10
- xgboost >= 2.0.0（GPU 加速需 CUDA 支持）
- scikit-learn >= 1.8.0
- joblib, numpy, pandas

### 复现基线结果（RandomForest）

```bash
# 下载数据集
python dataset/download_wine_data.py

# 训练 RandomForest 模型
python src/train_random_forest.py

# 预测评估
python src/predict_random_forest.py
```

### 复现优化结果（XGBoost GPU）

```bash
# 下载数据集
python dataset/download_wine_data.py

# 训练 XGBoost GPU 模型（自动执行特征工程 + 超参数搜索 + 早停）
python src/train.py

# 预测评估
python src/predict.py
```

## 📈 模型性能

### 整体表现

| 质量等级 | 基线 F1 | 优化 F1 | 基线 Recall | 优化 Recall | 变化 |
|:--------:|:-------:|:-------:|:-----------:|:-----------:|:----:|
| 3 | 0.00 | 0.00 | 0.00 | 0.00 | = |
| **4** | **0.00** | **0.29** | **0.00** | **0.30** | 🚀 |
| 5 | 0.67 | **0.69** | 0.64 | **0.69** | ⬆️ |
| 6 | 0.67 | 0.65 | 0.77 | 0.62 | ⬇️ |
| **7** | **0.52** | **0.61** | **0.44** | **0.69** | 🚀 |
| **8** | **0.23** | **0.47** | **0.13** | **0.49** | 🚀 |
| 9 | 0.00 | 0.00 | 0.00 | 0.00 | = |

### 关键改进亮点

1. **类别 4（原完全无法预测）**：F1 从 0.00 提升至 0.29，precision=0.27, recall=0.30
2. **类别 8（稀有类别）**：召回率从 13% 大幅提升至 49%，F1 从 0.23 提升至 0.47
3. **类别 7**：召回率从 44% 提升至 69%，F1 从 0.52 提升至 0.61
4. **宏平均 F1 跃升 30%**：从 0.30 提升至 0.39，表明模型在各类别间的表现更均衡

## 🖥️ 实验环境

| 项目 | 版本 |
|------|------|
| Python | 3.14.0 |
| scikit-learn | 1.8.0 |
| xgboost | 2.1.4 |
| joblib | 1.5.3 |
| numpy | 1.26.4 |
| GPU | NVIDIA H20 (CUDA 12.8, 96GB) |
| 随机种子 | 42 |

## 📚 项目结构

```
simple_cnn_test/
├── README.md               # 项目说明文档
├── baseline.md             # 基线测试与优化对比文档
├── answer_review/          # 测试结果评审
│   └── 第1轮测试结果.md
├── dataset/
│   ├── winequality.csv     # 合并的红酒+白酒数据集
│   ├── winequality-red.csv # 红酒数据
│   ├── winequality-white.csv # 白酒数据
│   └── download_wine_data.py # 数据下载脚本
├── src/
│   ├── train.py            # XGBoost GPU 训练脚本
│   └── predict.py          # XGBoost GPU 预测脚本
├── models/
│   ├── wine_model.joblib   # 训练好的模型
│   ├── train_data.joblib   # 训练集数据
│   └── test_data.joblib    # 测试集数据
└── requirements.txt
```

## 📄 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE)。
