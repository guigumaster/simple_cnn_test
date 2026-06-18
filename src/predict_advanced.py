#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测脚本：加载训练好的 XGBoost + Optuna 集成模型进行红酒质量预测。

使用 Top-N XGBoost multi:softprob 集成策略：
  1. 加载 N 个训练好的 XGBoost multi:softprob 模型
  2. 每个模型输出 7 类概率（quality 3-9 对应 0-6）
  3. 所有模型概率取平均
  4. 取 argmax 得到最终预测
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

# ── 项目路径 ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ordinal_ensemble_model.joblib")
TEST_DATA_PATH = os.path.join(MODEL_DIR, "ordinal_test_data.joblib")

# ── 质量等级定义 ────────────────────────────────────────────────────
QUALITY_LEVELS = [3, 4, 5, 6, 7, 8, 9]
N_CLASSES = len(QUALITY_LEVELS)

# ── 标签映射 ────────────────────────────────────────────────────────
LABEL_MAP = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

BASE_FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "wine_type",
]

TARGET_COLUMN = "quality"

LOG_TRANSFORM_COLS = [
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
]


def feature_engineering(df):
    """
    增强特征工程（与 train_advanced.py 完全一致）。
    输入 DataFrame，输出 43 维增强特征矩阵。
    """
    base = df[BASE_FEATURE_COLUMNS].values.copy()
    eps = 1e-8

    # --- 对数变换 ---
    log_feats = []
    for c in LOG_TRANSFORM_COLS:
        log_feats.append(np.log1p(df[c].values))
    log_data = np.column_stack(log_feats)

    # --- 酒类交互特征 ---
    wt = df["wine_type"].values
    interact_cols = [
        "alcohol", "volatile acidity", "residual sugar",
        "sulphates", "free sulfur dioxide", "total sulfur dioxide", "density",
    ]
    interact_data = [df[c].values * wt for c in interact_cols]
    interactions = np.column_stack(interact_data)

    # --- 比例/组合特征 ---
    so2_ratio = df["free sulfur dioxide"].values / (df["total sulfur dioxide"].values + eps)
    alc_sugar_ratio = df["alcohol"].values / (df["residual sugar"].values + 1.0)
    vol_alc = df["volatile acidity"].values * df["alcohol"].values
    citric_volatile = df["citric acid"].values / (df["volatile acidity"].values + eps)
    sulph_alc = df["sulphates"].values * df["alcohol"].values
    ph_alc = df["pH"].values * df["alcohol"].values
    total_acidity = df["fixed acidity"].values + df["volatile acidity"].values
    free_so2_pct = df["free sulfur dioxide"].values / (df["total sulfur dioxide"].values + eps)
    citric_fixed_ratio = df["citric acid"].values / (df["fixed acidity"].values + eps)

    ratio_features = np.column_stack([
        so2_ratio, alc_sugar_ratio, vol_alc,
        citric_volatile, sulph_alc, ph_alc,
        total_acidity, free_so2_pct, citric_fixed_ratio,
    ])

    # --- 平方项 ---
    square_cols = ["alcohol", "volatile acidity", "density", "residual sugar"]
    square_data = [df[c].values ** 2 for c in square_cols]
    squares = np.column_stack(square_data)

    # --- 额外多项式交互特征 ---
    acid_balance = (df["citric acid"].values + eps) / (df["pH"].values + eps)
    volatile_ph = df["volatile acidity"].values * df["pH"].values
    alcohol_density = df["alcohol"].values / (df["density"].values + eps)
    sulphate_volatile = df["sulphates"].values / (df["volatile acidity"].values + eps)
    chlorides_sugar = df["chlorides"].values * df["residual sugar"].values
    sqrt_total_so2 = np.sqrt(df["total sulfur dioxide"].values + eps)
    free_so2_wt = df["free sulfur dioxide"].values * wt

    extra_features = np.column_stack([
        acid_balance, volatile_ph, alcohol_density,
        sulphate_volatile, chlorides_sugar, sqrt_total_so2, free_so2_wt,
    ])

    # 合并
    X_enhanced = np.column_stack([
        base, log_data, interactions, ratio_features,
        squares, extra_features
    ])

    return X_enhanced


def load_ensemble_model(model_path):
    """
    加载训练好的 XGBoost 集成模型包

    Args:
        model_path: 模型文件路径

    Returns:
        model_package: 包含所有模型、标准化器、参数的字典
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"模型文件不存在：{model_path}\n"
            f"请先运行训练脚本: python src/train_advanced.py"
        )

    model_package = joblib.load(model_path)
    print(f"模型已加载：{model_path}")

    n_models = len(model_package.get("models", []))
    print(f"  集成模型数量: {n_models}")
    print(f"  最优参数: {model_package.get('best_params', {})}")

    return model_package


def ensemble_predict(model_package, X):
    """
    使用 Top-N 集成模型进行预测。

    Args:
        model_package: 模型包字典
        X: 输入特征矩阵

    Returns:
        predictions: 预测 quality 值 (3-9)
        probabilities: 每个类别的平均概率 (n_samples, 7)
        confidences: 预测置信度
    """
    models = model_package["models"]
    scaler = model_package["scaler"]

    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # 标准化
    X_scaled = scaler.transform(X)

    n_samples = X.shape[0]
    n_models = len(models)
    all_probas = np.zeros((n_samples, N_CLASSES, n_models))

    for i, model in enumerate(models):
        proba = model.predict_proba(X_scaled)
        all_probas[:, :, i] = proba

    # 平均概率
    avg_probas = np.mean(all_probas, axis=2)

    # 预测（映射标签 0-6 → 原始 quality 3-9）
    pred_mapped = np.argmax(avg_probas, axis=1)
    predictions = np.array([INV_LABEL_MAP[int(v)] for v in pred_mapped])

    # 置信度：最大概率值
    confidences = np.max(avg_probas, axis=1)

    return predictions, avg_probas, confidences


def print_feature_description(feature_values):
    """
    打印基础特征描述（12个基础特征）

    Args:
        feature_values: 特征值数组（43维，前12个是基础特征）
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
        ("酒类型", "wine_type"),
    ]

    print("   基础特征详情:")
    for (cn, en), val in zip(feature_descriptions, feature_values):
        if en == "wine_type":
            label = "红" if val == 0 else "白"
            print(f"   {cn} ({en}): {int(val)} ({label})")
        else:
            print(f"   {cn} ({en}): {val:.4f}")


def print_top_3_probs(probas):
    """打印概率最高的 3 个质量等级"""
    probas = np.array(probas)
    top_3_idx = np.argsort(probas)[-3:][::-1]
    quality_order = sorted(INV_LABEL_MAP.keys())  # 0,1,2,3,4,5,6
    top_3_str = ", ".join(
        f"{INV_LABEL_MAP[quality_order[j]]}({probas[j]:.3f})"
        for j in top_3_idx
    )
    print(f"   最可能的 3 个质量等级：{top_3_str}")


def main():
    """主函数 - 使用训练好的集成模型进行红酒质量预测"""
    print("=" * 70)
    print("  红酒质量预测 - XGBoost multi:softprob + Optuna 集成预测")
    print(f"  Top-N 集成策略 + 43 维增强特征")
    print("=" * 70)

    # 1. 加载模型
    print(f"\n[1/3] 加载集成模型...")
    model_package = load_ensemble_model(MODEL_PATH)

    # 2. 加载测试数据
    print(f"\n[2/3] 加载测试数据集...")
    if not os.path.exists(TEST_DATA_PATH):
        print(f"  测试数据集不存在: {TEST_DATA_PATH}")
        print(f"  请先运行训练脚本: python src/train_advanced.py")
        return

    test_package = joblib.load(TEST_DATA_PATH)
    X_test = test_package["X_test"]
    y_test = test_package["y_test"]
    print(f"  测试集样本数: {len(y_test)}")

    target_names = [str(l) for l in sorted(set(y_test))]
    print(f"  质量等级: {', '.join(target_names)}")

    # 各类别分布
    print(f"\n  测试集各类别样本分布:")
    for label in target_names:
        count = int((y_test == int(label)).sum())
        print(f"    质量 {label}: {count:5d} 个样本 ({count/len(y_test):.2%})")

    # 3. 预测
    print(f"\n[3/3] 进行集成预测...")
    predictions, avg_probas, confidences = ensemble_predict(
        model_package, X_test
    )

    print(f"\n预测结果:")
    print("=" * 70)

    # 最多显示 20 个样本的详细信息
    max_display = min(20, len(y_test))

    for i in range(max_display):
        pred = predictions[i]
        true = y_test[i]
        status = "[OK]" if pred == true else "[ERR]"

        print(f"\n样本 {i+1}: 预测质量={pred}, 真实质量={true} {status}")

        # 基础特征详情
        print_feature_description(X_test[i, :12])

        # Top-3 概率
        print_top_3_probs(avg_probas[i])
        print(f"    置信度: {confidences[i]:.4f}")

    # 总体准确率
    correct = int((predictions == y_test).sum())
    accuracy = correct / len(y_test)

    print(f"\n{'='*70}")
    print(f"  测试集准确率: {accuracy:.2%} ({correct}/{len(y_test)})")
    print(f"{'='*70}")

    # 详细分类报告
    unique_labels = sorted(set(y_test))
    print(f"\n分类评估报告:")
    print(classification_report(
        y_test, predictions, labels=unique_labels,
        target_names=[str(l) for l in unique_labels]
    ))

    # 混淆矩阵
    cm = confusion_matrix(y_test, predictions, labels=unique_labels)
    print(f"\n混淆矩阵（行=真实, 列=预测）:")
    header = "      " + "  ".join(f"{l:4d}" for l in unique_labels)
    print(header)
    for i, label in enumerate(unique_labels):
        row = "  ".join(f"{v:4d}" for v in cm[i])
        print(f"  {label}:  {row}")

    # 各类别召回率
    print(f"\n各类别召回率:")
    for i, label in enumerate(unique_labels):
        total = cm[i].sum()
        recall = cm[i, i] / total if total > 0 else 0.0
        print(f"  质量 {label}: {recall:.4f} ({cm[i, i]}/{total})")

    return predictions, avg_probas


if __name__ == "__main__":
    main()