#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测脚本：加载训练好的 XGBoost 模型与测试数据集进行红酒质量预测。
测试数据集由 train.py 训练时保存，确保训练/测试集明确分离。

标签映射说明：
  XGBoost multi:softprob 要求类别从 0 开始，因此训练时将 quality 3-9
  映射为 0-6。predict.py 加载模型后，预测输出会自动映射回原始 quality 值。
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ── 项目路径 ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
TEST_DATA_PATH = os.path.join(MODEL_DIR, "test_data.joblib")

# ── 基础特征列（与 train.py 保持一致）────────────────────────────────
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

# ── 标签映射（与 train.py 完全一致）─────────────────────────────────
LABEL_MAP = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# ── 对数变换列（与 train.py 完全一致）────────────────────────────────
LOG_TRANSFORM_COLS = [
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
]


def feature_engineering(df):
    """
    特征工程（与 train.py 中的实现完全一致）。
    输入 DataFrame，输出增强后的特征矩阵。
    """
    # --- 1. 基础特征 ---
    base = df[BASE_FEATURE_COLUMNS].values.copy()

    # --- 2. 对数变换特征 ---
    log_feats = []
    for c in LOG_TRANSFORM_COLS:
        log_feats.append(np.log1p(df[c].values))
    log_data = np.column_stack(log_feats)

    # --- 3. 酒类交互特征 ---
    wt = df["wine_type"].values
    _interact_cols = [
        "alcohol", "volatile acidity", "residual sugar",
        "sulphates", "free sulfur dioxide", "total sulfur dioxide", "density",
    ]
    _interact_data = [df[c].values * wt for c in _interact_cols]
    interactions = np.column_stack(_interact_data)

    # --- 4. 比例 / 组合特征 ---
    eps = 1e-8
    so2_ratio = df["free sulfur dioxide"].values / (df["total sulfur dioxide"].values + eps)
    alc_sugar_ratio = df["alcohol"].values / (df["residual sugar"].values + 1.0)
    vol_alc = df["volatile acidity"].values * df["alcohol"].values
    citric_volatile = df["citric acid"].values / (df["volatile acidity"].values + eps)
    sulph_alc = df["sulphates"].values * df["alcohol"].values
    ph_alc = df["pH"].values * df["alcohol"].values
    # 新增特征
    total_acidity = df["fixed acidity"].values + df["volatile acidity"].values
    free_so2_pct = df["free sulfur dioxide"].values / (df["total sulfur dioxide"].values + eps)
    citric_fixed_ratio = df["citric acid"].values / (df["fixed acidity"].values + eps)

    ratio_features = np.column_stack([
        so2_ratio, alc_sugar_ratio, vol_alc,
        citric_volatile, sulph_alc, ph_alc,
        total_acidity, free_so2_pct, citric_fixed_ratio,
    ])

    # --- 5. 平方项 ---
    _square_cols = ["alcohol", "volatile acidity", "density", "residual sugar"]
    _square_data = [df[c].values ** 2 for c in _square_cols]
    squares = np.column_stack(_square_data)

    # 合并
    X_enhanced = np.column_stack([base, log_data, interactions, ratio_features, squares])
    return X_enhanced


def load_model(model_path):
    """
    加载已保存的模型

    Args:
        model_path: 模型文件路径

    Returns:
        model: 加载的 XGBoost 模型
        scaler: 加载的标准化器
        best_params: 最优超参数（可选）
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在：{model_path}")

    checkpoint = joblib.load(model_path)
    model = checkpoint["model"]
    scaler = checkpoint.get("scaler")
    best_params = checkpoint.get("best_params", None)

    print(f"模型已加载：{model_path}")
    if best_params:
        print(f"超参数：{best_params}")

    return model, scaler


def predict(model, scaler, X):
    """
    使用模型进行预测（自动将映射后的标签转换回原始 quality 值）

    Args:
        model: 训练好的 XGBoost 模型
        scaler: 标准化器
        X: 输入特征矩阵

    Returns:
        predictions: 预测结果（原始 quality 值：3-9）
        probabilities: 预测概率（列对应映射后标签 0-6，索引与原始 quality 对应）
    """
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    X_scaled = scaler.transform(X)

    # 预测的映射标签（0-6）
    pred_mapped = model.predict(X_scaled)

    # 将映射标签（0-6）转换回原始 quality（3-9）
    predictions = np.array([INV_LABEL_MAP[int(v)] for v in pred_mapped])

    # 预测概率（列对应映射标签 0-6）
    prob_mapped = model.predict_proba(X_scaled) if hasattr(model, "predict_proba") else None

    # 将概率列重新排序为原始 quality 顺序
    if prob_mapped is not None:
        n_classes = prob_mapped.shape[1]
        # 映射标签顺序：[0,1,2,3,4,5,6] → [3,4,5,6,7,8,9]
        orig_order = [INV_LABEL_MAP[i] for i in range(n_classes)]
        probabilities = {"probabilities": prob_mapped, "class_order": orig_order}
    else:
        probabilities = None

    return predictions, probabilities


def print_feature_description(feature_values):
    """
    打印基础特征描述（仅打印 12 个基础特征）

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
        ("酒类型", "wine_type"),
    ]

    print("   基础特征详情:")
    for (cn, en), val in zip(feature_descriptions, feature_values):
        if en == "wine_type":
            label = "红" if val == 0 else "白"
            print(f"   {cn} ({en}): {int(val)} ({label})")
        else:
            print(f"   {cn} ({en}): {val:.4f}")


def main():
    """主函数 - 使用独立测试集进行预测"""
    print("=" * 60)
    print("红酒质量预测任务 - XGBoost GPU 预测脚本 v2.0")
    print("使用独立测试数据集进行评估")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/3] 加载模型...")
    model_path = os.path.join(MODEL_DIR, "wine_model.joblib")
    model, scaler = load_model(model_path)

    # 2. 加载测试数据集
    print("\n[2/3] 加载测试数据集...")
    if not os.path.exists(TEST_DATA_PATH):
        print(f"错误：测试数据集不存在 {TEST_DATA_PATH}")
        print("请先运行训练脚本：python src/train.py")
        return

    test_data = joblib.load(TEST_DATA_PATH)
    X_test = test_data["X"]
    y_test = test_data["y"]
    test_indices = test_data["indices"]

    print(f"测试集样本数：{len(y_test)}")
    print(f"测试集索引范围：{test_indices[0]}~{test_indices[-1]} (共 {len(test_indices)} 个)")

    target_names = [str(i) for i in sorted(np.unique(y_test))]
    print(f"质量等级：{', '.join(target_names)}")

    # 各类别分布
    print("\n测试集各类别样本分布:")
    for label in target_names:
        count = np.sum(y_test == int(label))
        print(f"  质量 {label}: {count:5d} 个样本 ({count/len(y_test):.2%})")

    # 3. 进行预测
    print("\n[3/3] 进行预测...")
    predictions, prob_info = predict(model, scaler, X_test)

    print("\n预测结果:")
    print("=" * 60)

    correct_count = 0
    # 最多显示 20 个样本的详细信息
    max_display = min(20, len(y_test))

    for i in range(max_display):
        pred = predictions[i]
        true = y_test[i]
        status = "[OK]" if pred == true else "[ERR]"
        if pred == true:
            correct_count += 1

        print(f"\n样本 {i+1} (原始索引={test_indices[i]}): "
              f"预测质量={pred}, 真实质量={true} {status}")

        # 基础特征值
        print_feature_description(X_test[i, :12])

        # 显示概率最高的 3 个类别
        if prob_info is not None:
            probs = prob_info["probabilities"][i]
            class_order = prob_info["class_order"]
            top_3_idx = np.argsort(probs)[-3:][::-1]
            top_3_str = ", ".join(
                f"{class_order[j]}({probs[j]:.3f})" for j in top_3_idx
            )
            print(f"   最可能的 3 个质量等级：{top_3_str}")

    # 计算总体准确率
    total_correct = np.sum(predictions == y_test)
    accuracy = total_correct / len(y_test)

    print("\n" + "=" * 60)
    print(f"测试集准确率：{accuracy:.2%} ({total_correct}/{len(y_test)})")
    print("=" * 60)

    # 详细分类报告
    print("\n详细分类报告:")
    from sklearn.metrics import classification_report, confusion_matrix
    unique_labels = sorted(set(y_test))
    print(classification_report(y_test, predictions, labels=unique_labels,
                                target_names=[str(l) for l in unique_labels]))

    # 混淆矩阵
    cm = confusion_matrix(y_test, predictions, labels=unique_labels)
    print("\n混淆矩阵（行=真实, 列=预测）:")
    print("      " + "  ".join(f"{l:4d}" for l in unique_labels))
    for i, label in enumerate(unique_labels):
        row = "  ".join(f"{v:4d}" for v in cm[i])
        print(f"  {label}:  {row}")

    return predictions, prob_info


if __name__ == "__main__":
    main()