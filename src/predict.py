#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测脚本：加载训练好的多策略融合模型进行红酒质量预测
支持交互特征工程、GPU XGBoost 和独立测试集评估
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


# ========== 路径配置 ==========
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'wine_model.joblib')
TEST_DATA_PATH = os.path.join(MODEL_DIR, 'test_data.joblib')

# 特征列名（与 train.py 保持一致）
ALL_FEATURES = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol',
    'wine_type',
]

# ========== 质量标签映射 ==========
QUALITY_MAP = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}
QUALITY_INV_MAP = {v: k for k, v in QUALITY_MAP.items()}
QUALITY_LABELS = ['3', '4', '5', '6', '7', '8', '9']


def load_model(model_path=MODEL_PATH):
    """
    加载训练好的模型及所有预处理组件

    Returns:
        model: XGBoost 模型
        scaler: 标准化器
        poly: PolynomialFeatures 交互特征生成器
        feature_columns: 特征列名
        class_weight_dict: 类别权重字典
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"模型文件不存在：{model_path}\n"
            f"请先运行训练脚本：python {os.path.join(PROJECT_ROOT, 'src', 'train.py')}"
        )

    checkpoint = joblib.load(model_path)
    model = checkpoint['model']
    scaler = checkpoint['scaler']
    poly = checkpoint['poly']
    feature_columns = checkpoint.get('feature_columns', ALL_FEATURES)
    class_weight_dict = checkpoint.get('class_weight_dict', None)
    quality_map = checkpoint.get('quality_map', QUALITY_MAP)
    quality_inv_map = checkpoint.get('quality_inv_map', QUALITY_INV_MAP)
    use_interaction = checkpoint.get('use_interaction', True)

    print(f"模型已加载：{model_path}")
    print(f"模型类型：{type(model).__name__}")
    if poly is not None:
        print(f"交互特征维度：{poly.n_output_features_ if hasattr(poly, 'n_output_features_') else 'N/A'}")
    else:
        print(f"基础特征维度：{len(feature_columns)}")
    print(f"质量标签映射：{dict(quality_inv_map)}")

    return model, scaler, poly, feature_columns, class_weight_dict, quality_map, quality_inv_map, use_interaction


def predict(model, scaler, poly, X_raw, quality_inv_map=None):
    """
    使用完整预处理管线进行预测

    Args:
        model: XGBoost 模型
        scaler: 标准化器
        poly: PolynomialFeatures 对象（可为 None，表示不使用交互特征）
        X_raw: 原始输入特征（numpy array 或 list, 形状为 (n, n_features) 或 (n_features,)）
        quality_inv_map: 逆映射字典，将 0-indexed 预测映射回原始标签

    Returns:
        predictions: 预测类别（原始标签）
        probabilities: 各类别预测概率
    """
    if quality_inv_map is None:
        quality_inv_map = QUALITY_INV_MAP

    X_raw = np.array(X_raw, dtype=np.float64)
    if X_raw.ndim == 1:
        X_raw = X_raw.reshape(1, -1)

    # 构建 DataFrame
    n_base_features = len(ALL_FEATURES)
    if X_raw.shape[1] != n_base_features:
        raise ValueError(
            f"输入特征数量不匹配：期望 {n_base_features} 个特征（11 理化 + wine_type），"
            f"实际传入 {X_raw.shape[1]} 个"
        )

    X_df = pd.DataFrame(X_raw, columns=ALL_FEATURES[:X_raw.shape[1]])

    # Step 1: 特征变换（有交互 / 无交互）
    if poly is not None:
        X_transformed = poly.transform(X_df)
    else:
        X_transformed = X_df.values

    # Step 2: 标准化
    X_scaled = scaler.transform(X_transformed)

    # Step 3: 预测（0-indexed）
    pred_mapped = model.predict(X_scaled)

    # Step 4: 映射回原始标签
    predictions = np.array([quality_inv_map[int(p)] for p in pred_mapped])

    # Step 5: 预测概率
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_scaled)

    return predictions, probabilities


def print_feature_description(feature_values, feature_names=ALL_FEATURES):
    """打印特征描述"""
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
        ("酒类型", "wine_type (0=红, 1=白)"),
    ]

    print("   特征详情:")
    for (cn, en), val in zip(feature_descriptions, feature_values):
        if en.startswith("wine_type"):
            wine_str = "红酒" if val < 0.5 else "白酒"
            print(f"   {cn} ({en}): {val:.4f}  [{wine_str}]")
        else:
            print(f"   {cn} ({en}): {val:.4f}")


def evaluate_on_test_set(model, scaler, poly, quality_inv_map=None):
    """
    在独立测试集上评估模型性能

    Args:
        model: XGBoost 模型
        scaler: 标准化器
        poly: PolynomialFeatures 对象（可为 None）
        quality_inv_map: 逆映射字典，将 0-indexed 标签映射回原始标签

    Returns:
        accuracy: 准确率（基于原始标签）
        y_test: 真实标签（原始标签）
        y_pred: 预测标签（原始标签）
    """
    if quality_inv_map is None:
        quality_inv_map = QUALITY_INV_MAP

    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(
            f"测试数据集不存在：{TEST_DATA_PATH}\n"
            f"请先运行训练脚本：python {os.path.join(PROJECT_ROOT, 'src', 'train.py')}"
        )

    test_data = joblib.load(TEST_DATA_PATH)
    X_test = test_data['X']
    y_test_mapped = test_data['y']  # 0-indexed 标签
    test_indices = test_data['indices']

    # 转为 DataFrame
    feature_cols = test_data.get('feature_columns', ALL_FEATURES[:X_test.shape[1]])
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)

    # 特征变换（有交互 / 无交互）
    if poly is not None:
        X_test_transformed = poly.transform(X_test_df)
    else:
        X_test_transformed = X_test_df.values

    # 标准化
    X_test_scaled = scaler.transform(X_test_transformed)
    # 预测（0-indexed）
    y_pred_mapped = model.predict(X_test_scaled)

    # 映射回原始标签
    y_test = np.array([quality_inv_map[int(t)] for t in y_test_mapped])
    y_pred = np.array([quality_inv_map[int(p)] for p in y_pred_mapped])

    accuracy = accuracy_score(y_test, y_pred)

    # 分类报告
    target_names = sorted(set(str(l) for l in np.unique(y_test)))
    print("\n" + "=" * 70)
    print("独立测试集评估报告")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    # 综合指标
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"准确率 (Accuracy):      {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"宏平均 F1 (Macro F1):  {macro_f1:.4f}")
    print(f"加权平均 F1 (Weighted): {weighted_f1:.4f}")

    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred, labels=sorted(np.unique(y_test)))
    unique_labels = sorted(np.unique(y_test))
    header = "         " + "  ".join(f"{l:>5}" for l in unique_labels)
    print("预测值:")
    print(header)
    print("真实值:")
    for i, label in enumerate(unique_labels):
        row = f"  {label:>2}:    " + "  ".join(f"{v:>5}" for v in cm[i])
        print(row)

    return accuracy, y_test, y_pred


def predict_single_sample(model, scaler, poly, sample_features=None, quality_inv_map=None, use_interaction=True):
    """
    预测单个样本（交互模式）

    Args:
        model: XGBoost 模型
        scaler: 标准化器
        poly: PolynomialFeatures 对象
        sample_features: 可选，直接传入特征值
    """
    print("\n[单样本预测模式]")
    print("=" * 60)

    if sample_features is not None:
        features = np.array(sample_features, dtype=np.float64)
    else:
        # 用一个中等红酒样本作为示例
        features = np.array([
            7.4,    # fixed acidity
            0.70,   # volatile acidity
            0.00,   # citric acid
            1.9,    # residual sugar
            0.076,  # chlorides
            11.0,   # free sulfur dioxide
            34.0,   # total sulfur dioxide
            0.9978, # density
            3.51,   # pH
            0.56,   # sulphates
            9.4,    # alcohol
            0.0,    # wine_type (0=红酒)
        ])

    if quality_inv_map is None:
        quality_inv_map = QUALITY_INV_MAP
    predictions, probabilities = predict(model, scaler, poly, features, quality_inv_map=quality_inv_map)
    n_features_used = len(features)

    print(f"输入特征 (wine_type={features[-1]:.0f} - {'红酒' if features[-1] < 0.5 else '白酒'})")
    print_feature_description(features)

    pred_label = int(predictions[0])
    print(f"\n>> 预测质量等级：{pred_label}")

    if probabilities is not None:
        prob = probabilities[0]
        # 模型类别（0-indexed）
        model_classes = model.classes_
        # 按概率排序
        top_indices = np.argsort(prob)[::-1]
        print(">> 各类别概率：")
        for idx in top_indices:
            cls_mapped = model_classes[idx]
            cls_orig = quality_inv_map.get(int(cls_mapped), cls_mapped)
            pct = prob[idx] * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"   质量 {cls_orig}: {pct:5.2f}% {bar}")

    return predictions, probabilities


def main():
    """主函数"""
    print("=" * 70)
    print("🍷 红酒质量预测 - 多策略融合模型推理")
    print("   模型：GPU XGBoost + wine_type + SMOTE + 交互特征")
    print("=" * 70)

    # 1. 加载模型
    print("\n[1/4] 加载模型及预处理组件...")
    try:
        model, scaler, poly, feature_columns, class_weight_dict, quality_map, quality_inv_map, use_interaction = load_model()
    except FileNotFoundError as e:
        print(f"错误：{e}")
        sys.exit(1)

    # 2. 在独立测试集上评估
    print("\n[2/4] 在独立测试集上评估...")
    try:
        accuracy, y_test, y_pred = evaluate_on_test_set(model, scaler, poly, quality_inv_map=quality_inv_map)
        print(f"\n>> 测试集准确率：{accuracy:.2%}")
    except FileNotFoundError as e:
        print(f"错误：{e}")
        print("跳过测试集评估...")

    # 3. 单样本预测演示
    print("\n[3/4] 单样本预测演示...")
    predict_single_sample(model, scaler, poly, quality_inv_map=quality_inv_map, use_interaction=use_interaction)

    # 4. 预测全部测试样本（详细输出）
    print("\n[4/4] 逐样本预测结果...")
    if os.path.exists(TEST_DATA_PATH):
        test_data = joblib.load(TEST_DATA_PATH)
        X_test_orig = test_data['X']
        y_test_mapped = test_data['y']  # 存储的是映射后的标签
        test_indices = test_data['indices']

        # 批量处理
        feature_cols = test_data.get('feature_columns', ALL_FEATURES[:X_test_orig.shape[1]])
        X_test_df = pd.DataFrame(X_test_orig, columns=feature_cols)
        if poly is not None:
            X_test_transformed = poly.transform(X_test_df)
        else:
            X_test_transformed = X_test_df.values
        X_test_scaled = scaler.transform(X_test_transformed)
        all_preds_mapped = model.predict(X_test_scaled)
        all_probs = model.predict_proba(X_test_scaled)

        # 映射回原始标签
        all_preds = np.array([quality_inv_map[int(p)] for p in all_preds_mapped])
        y_test = np.array([quality_inv_map[int(t)] for t in y_test_mapped])

        correct_count = 0
        sample_count = min(10, len(y_test))

        print(f"\n显示前 {sample_count} 个样本的预测详情:")
        print("-" * 70)
        for i in range(sample_count):
            pred = all_preds[i]
            true = y_test[i]
            prob = all_probs[i]
            status = "✓" if pred == true else "✗"
            if pred == true:
                correct_count += 1

            print(f"\n样本 {i+1} (原始索引={test_indices[i]}): "
                  f"预测={pred}, 真实={true} {status}")
            print_feature_description(X_test_orig[i], feature_cols)

            top_idx = np.argsort(prob)[-3:][::-1]
            top_labels = [str(quality_inv_map.get(int(j), j)) for j in model.classes_[top_idx]]
            top_probs = [prob[j] for j in top_idx]
            print(f"   Top-3: {top_labels[0]}({top_probs[0]:.2%}), "
                  f"{top_labels[1]}({top_probs[1]:.2%}), "
                  f"{top_labels[2]}({top_probs[2]:.2%})")

        if len(y_test) > 0:
            subset_acc = np.mean(all_preds == y_test)
            print(f"\n完整测试集准确率：{subset_acc:.2%} ({np.sum(all_preds == y_test)}/{len(y_test)})")

    print("\n" + "=" * 70)
    print("预测完成!")
    print("=" * 70)

    return model, scaler, poly


if __name__ == "__main__":
    main()