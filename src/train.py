#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本：使用红酒质量数据集训练分类模型
数据集：UCI Wine Quality（11 个理化特征，7 个质量等级）
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# 数据集路径
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'winequality.csv')

# 特征列名
FEATURE_COLUMNS = [
    'fixed acidity',       # 固定酸度
    'volatile acidity',    # 挥发性酸度
    'citric acid',         # 柠檬酸
    'residual sugar',      # 残留糖分
    'chlorides',           # 氯化物
    'free sulfur dioxide', # 游离二氧化硫
    'total sulfur dioxide',# 总二氧化硫
    'density',             # 密度
    'pH',                  # pH 值
    'sulphates',           # 硫酸盐
    'alcohol'              # 酒精含量
]

# 目标列名
TARGET_COLUMN = 'quality'

# 质量等级映射（用于报告）
QUALITY_LABELS = ['3', '4', '5', '6', '7', '8', '9']


def load_data():
    """
    加载红酒质量数据集

    Returns:
        X: 特征数据（11 个理化特征）
        y: 标签数据（质量等级 3-9）
        feature_names: 特征名称
        target_names: 类别名称
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"数据集不存在：{DATASET_PATH}\n请先运行 dataset/download_wine_data.py 下载数据")
    
    df = pd.read_csv(DATASET_PATH)
    
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    
    feature_names = FEATURE_COLUMNS
    # 只包含数据中实际存在的质量等级
    unique_classes = sorted(df[TARGET_COLUMN].unique())
    target_names = [str(i) for i in unique_classes]
    
    return X, y, feature_names, target_names


def train_model(X, y, test_size=0.2, random_state=42):
    """
    训练随机森林分类模型

    Args:
        X: 特征数据
        y: 标签数据
        test_size: 测试集比例
        random_state: 随机种子

    Returns:
        model: 训练好的模型
        scaler: 标准化器
        metrics: 评估指标字典
        train_idx: 训练集索引
        test_idx: 测试集索引
    """
    # 划分训练集和测试集，返回索引
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, np.arange(len(y)), test_size=test_size, random_state=random_state, stratify=y
    )

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 创建并训练随机森林模型
    # 使用适中的参数来避免过拟合，使准确率不会过高
    model = RandomForestClassifier(
        n_estimators=100,        # 树的数量（适中）
        max_depth=10,            # 最大深度（限制）
        min_samples_split=5,     # 内部节点再划分所需最小样本数（增加）
        min_samples_leaf=2,      # 叶节点所需最小样本数（增加）
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # 在测试集上评估
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'y_test': y_test,
        'y_pred': y_pred,
        'X_test': X_test,
        'X_test_scaled': X_test_scaled,
        'X_train': X_train,
        'y_train': y_train
    }

    return model, scaler, metrics, train_idx, test_idx


def save_model(model, scaler, save_path):
    """
    保存模型到文件

    Args:
        model: 训练好的模型
        scaler: 标准化器
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler}, save_path)
    print(f"模型已保存到：{save_path}")


def save_data_split(X_train, X_test, y_train, y_test, train_idx, test_idx, save_dir):
    """
    保存训练集和测试集的划分结果

    Args:
        X_train: 训练集特征
        X_test: 测试集特征
        y_train: 训练集标签
        y_test: 测试集标签
        train_idx: 训练集索引
        test_idx: 测试集索引
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存训练集
    train_data = {
        'X': X_train,
        'y': y_train,
        'indices': train_idx,
        'feature_columns': FEATURE_COLUMNS,
        'target_column': TARGET_COLUMN
    }
    train_path = os.path.join(save_dir, 'train_data.joblib')
    joblib.dump(train_data, train_path)
    print(f"训练集已保存到：{train_path} (样本数：{len(y_train)})")
    
    # 保存测试集
    test_data = {
        'X': X_test,
        'y': y_test,
        'indices': test_idx,
        'feature_columns': FEATURE_COLUMNS,
        'target_column': TARGET_COLUMN
    }
    test_path = os.path.join(save_dir, 'test_data.joblib')
    joblib.dump(test_data, test_path)
    print(f"测试集已保存到：{test_path} (样本数：{len(y_test)})")
    
    return train_path, test_path


def print_evaluation_report(y_test, y_pred, target_names):
    """
    打印详细评估报告

    Args:
        y_test: 真实标签
        y_pred: 预测标签
        target_names: 类别名称
    """
    print("\n分类评估报告:")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("混淆矩阵:")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
    unique_labels = sorted(set(y_test))
    
    # 打印列标题
    print("预测值:")
    print("   ", "  ".join(f"{label:5s}" for label in [str(l) for l in unique_labels]))
    print("真实值:")
    for i, label in enumerate(unique_labels):
        print(f" {label}: ", "  ".join(f"{v:5d}" for v in cm[i]))


def main():
    """主函数"""
    print("=" * 60)
    print("红酒质量预测任务 - 训练脚本")
    print("数据集：UCI Wine Quality (11 个理化特征，7 个质量等级)")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    X, y, feature_names, target_names = load_data()
    print(f"数据形状：{X.shape}")
    print(f"样本数量：{len(y)}")
    print(f"特征数量：{X.shape[1]}")
    print(f"类别数量：{len(target_names)} (质量等级：{', '.join(target_names)})")

    # 显示各类别样本数量
    print("\n各类别样本分布:")
    for label in target_names:
        count = np.sum(y == int(label))
        print(f"  质量 {label}: {count} 个样本")

    # 2. 训练模型
    print("\n[2/4] 训练模型...")
    print("模型：随机森林分类器 (100 棵树，最大深度 10)")
    model, scaler, metrics, train_idx, test_idx = train_model(X, y)
    print(f"测试集准确率：{metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"训练集样本数：{len(train_idx)}, 测试集样本数：{len(test_idx)}")

    # 3. 打印详细评估报告
    print("\n[3/4] 模型评估...")
    print_evaluation_report(metrics['y_test'], metrics['y_pred'], target_names)

    # 4. 保存模型和数据划分
    print("\n[4/4] 保存模型和数据集...")
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    save_path = os.path.join(model_dir, 'wine_model.joblib')
    save_model(model, scaler, save_path)
    
    # 保存训练集和测试集的划分
    save_data_split(
        metrics['X_train'], metrics['X_test'],
        metrics['y_train'], metrics['y_test'],
        train_idx, test_idx,
        model_dir
    )

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

    return model, scaler, metrics


if __name__ == "__main__":
    main()
