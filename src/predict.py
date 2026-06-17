#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测脚本：加载训练好的模型和测试数据集进行红酒质量预测
测试数据集由 train.py 训练时保存，确保训练/测试集明确分离
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd


# 特征列名（与 train.py 保持一致）
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

# 测试数据集路径（由 train.py 保存）
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'test_data.joblib')


def load_model(model_path):
    """
    加载已保存的模型

    Args:
        model_path: 模型文件路径

    Returns:
        model: 加载的模型
        scaler: 加载的标准化器
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在：{model_path}")
    checkpoint = joblib.load(model_path)
    model = checkpoint['model']
    scaler = checkpoint['scaler']
    print(f"模型已加载：{model_path}")
    return model, scaler


def predict(model, scaler, X):
    """
    使用模型进行预测

    Args:
        model: 训练好的模型
        scaler: 标准化器
        X: 输入特征数据

    Returns:
        predictions: 预测结果
        probabilities: 预测概率（如果模型支持）
    """
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # 特征标准化
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)

    # 获取预测概率（如果模型支持）
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_scaled)

    return predictions, probabilities


def print_feature_description(feature_values):
    """
    打印特征描述

    Args:
        feature_values: 特征值数组
    """
    feature_descriptions = [
        ("固定酸度", "fixed acidity"),
        ("挥发性酸度", "volatile acidity"),
        ("柠檬酸", "citric acid"),
        ("残留糖分", "residual sugar"),
        ("氯化物", "chlorides"),
        ("游离二氧化硫", "free sulfur dioxide"),
        ("总二氧化硫", "total sulfur dioxide"),
        ("密度", "density"),
        ("pH 值", "pH"),
        ("硫酸盐", "sulphates"),
        ("酒精含量", "alcohol"),
    ]
    
    print("   特征详情:")
    for (cn, en), val in zip(feature_descriptions, feature_values):
        print(f"   {cn} ({en}): {val:.4f}")


def main():
    """主函数 - 使用独立测试集进行预测"""
    print("=" * 60)
    print("红酒质量预测任务 - 预测脚本")
    print("使用独立测试数据集进行评估")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/3] 加载模型...")
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(model_dir, 'wine_model.joblib')
    model, scaler = load_model(model_path)

    # 2. 加载测试数据集（由 train.py 保存的独立测试集）
    print("\n[2/3] 加载测试数据集...")
    if not os.path.exists(TEST_DATA_PATH):
        print(f"错误：测试数据集不存在 {TEST_DATA_PATH}")
        print("请先运行训练脚本：python src/train.py")
        return

    test_data = joblib.load(TEST_DATA_PATH)
    X_test = test_data['X']
    y_test = test_data['y']
    test_indices = test_data['indices']
    
    print(f"测试集样本数：{len(y_test)}")
    print(f"测试集索引：{test_indices[:10]}... (显示前 10 个)")
    target_names = [str(i) for i in sorted(np.unique(y_test))]
    print(f"质量等级：{', '.join(target_names)}")

    # 3. 进行预测
    print("\n[3/3] 进行预测...")
    predictions, probabilities = predict(model, scaler, X_test)

    print("\n预测结果:")
    print("=" * 60)

    correct_count = 0
    for i, (pred, true, prob, features) in enumerate(zip(predictions, y_test, probabilities, X_test)):
        status = "[OK]" if pred == true else "[ERR]"
        if pred == true:
            correct_count += 1

        print(f"\n样本 {i+1} (原始索引={test_indices[i]}): 预测质量={pred}, 真实质量={true} {status}")
        print_feature_description(features)

        # 显示前 3 个最高概率
        top_3_idx = np.argsort(prob)[-3:][::-1]
        top_3_probs = prob[top_3_idx]
        # 获取模型训练的类别标签
        model_classes = model.classes_
        top_3_labels = [str(model_classes[i]) for i in top_3_idx]
        print(f"   最可能的 3 个质量等级：{top_3_labels[0]}({top_3_probs[0]:.3f}), "
              f"{top_3_labels[1]}({top_3_probs[1]:.3f}), {top_3_labels[2]}({top_3_probs[2]:.3f})")

    print("\n" + "=" * 60)
    # 计算准确率
    accuracy = np.mean(predictions == y_test)
    print(f"\n测试集准确率：{accuracy:.2%} ({correct_count}/{len(predictions)})")

    return predictions, probabilities


if __name__ == "__main__":
    main()
