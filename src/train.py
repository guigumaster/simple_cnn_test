#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本：使用红酒质量数据集训练 XGBoost 分类模型（GPU 加速）
改进点：
  - 修复标签映射问题（quality 3-9 → 0-6，适配 XGBoost multi:softprob）
  - 加入被遗漏的 wine_type 特征
  - 引入有意义的交互特征 + 对数变换（特征工程，共 36 维）
  - 使用 XGBoost（GPU: tree_method=hist + device=cuda）加速训练
  - 样本权重处理类别不平衡
  - 早停机制 + 超参数调优
数据集：UCI Wine Quality（11 个理化特征 + wine_type，7 个质量等级）
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)

# ── 项目路径 ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "winequality.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ── 标签映射（XGBoost multi:softprob 要求类别从 0 开始）────────────
LABEL_MAP = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# ── 基础特征列（共 12 个：11 个理化特征 + wine_type）────────────────
BASE_FEATURE_COLUMNS = [
    "fixed acidity",           # 固定酸度
    "volatile acidity",        # 挥发性酸度
    "citric acid",             # 柠檬酸
    "residual sugar",          # 残留糖分
    "chlorides",               # 氯化物
    "free sulfur dioxide",     # 游离二氧化硫
    "total sulfur dioxide",    # 总二氧化硫
    "density",                 # 密度
    "pH",                      # pH 值
    "sulphates",               # 硫酸盐
    "alcohol",                 # 酒精含量
    "wine_type",               # 酒类型（0=红，1=白）
]

TARGET_COLUMN = "quality"

# ── 用于对数变换的偏态特征（取 log(1+x) 缓解长尾分布）───────────────
LOG_TRANSFORM_COLS = [
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
]


def feature_engineering(df):
    """
    在 DataFrame 上构建增强特征集。
    输入包含 BASE_FEATURE_COLUMNS 中的列，输出为 numpy 数组。
    返回 (feature_matrix, engineered_feature_names)
    """
    names = list(BASE_FEATURE_COLUMNS)

    # --- 1. 基础特征（原始值）---
    base = df[BASE_FEATURE_COLUMNS].values.copy()

    # --- 2. 对数变换特征（偏态特征取 log(1+x)）---
    log_feats = []
    log_names = []
    for c in LOG_TRANSFORM_COLS:
        log_feats.append(np.log1p(df[c].values))
        log_names.append(f"log_{c}")
    log_data = np.column_stack(log_feats)

    # --- 3. 酒类交互特征（wine_type × 关键理化指标）---
    wt = df["wine_type"].values
    _interact_cols = [
        "alcohol",
        "volatile acidity",
        "residual sugar",
        "sulphates",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
    ]
    _interact_data = [df[c].values * wt for c in _interact_cols]
    interactions = np.column_stack(_interact_data)
    _interact_names = [f"{c}_x_wine_type" for c in _interact_cols]

    # --- 4. 比例 / 组合特征 ---
    eps = 1e-8
    so2_ratio = (
        df["free sulfur dioxide"].values / (df["total sulfur dioxide"].values + eps)
    )
    alc_sugar_ratio = df["alcohol"].values / (df["residual sugar"].values + 1.0)
    vol_alc = df["volatile acidity"].values * df["alcohol"].values
    citric_volatile = df["citric acid"].values / (df["volatile acidity"].values + eps)
    sulph_alc = df["sulphates"].values * df["alcohol"].values
    ph_alc = df["pH"].values * df["alcohol"].values
    # 新增: 总酸度 = fixed acidity + volatile acidity
    total_acidity = df["fixed acidity"].values + df["volatile acidity"].values
    # 新增: 游离SO2占比
    free_so2_pct = df["free sulfur dioxide"].values / (df["total sulfur dioxide"].values + eps)
    # 新增: 酸度平衡 = citric acid / (fixed acidity + eps)
    citric_fixed_ratio = df["citric acid"].values / (df["fixed acidity"].values + eps)

    ratio_features = np.column_stack([
        so2_ratio, alc_sugar_ratio, vol_alc,
        citric_volatile, sulph_alc, ph_alc,
        total_acidity, free_so2_pct, citric_fixed_ratio,
    ])
    ratio_names = [
        "so2_ratio", "alc_sugar_ratio", "vol_alc_interact",
        "citric_volatile_ratio", "sulph_alc_interact", "ph_alc_interact",
        "total_acidity", "free_so2_pct", "citric_fixed_ratio",
    ]

    # --- 5. 平方项（捕捉非线性）---
    _square_cols = ["alcohol", "volatile acidity", "density", "residual sugar"]
    _square_data = [df[c].values ** 2 for c in _square_cols]
    squares = np.column_stack(_square_data)
    _square_names = [f"{c}_squared" for c in _square_cols]

    # --- 合并所有特征 ---
    X_enhanced = np.column_stack([base, log_data, interactions, ratio_features, squares])
    all_names = names + log_names + _interact_names + ratio_names + _square_names

    return X_enhanced, all_names


def compute_sample_weights(y):
    """
    计算样本权重：每个类别的权重 = 总样本数 / (类别数 × 该类样本数)
    使得稀有类别获得更高的权重。
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    class_weight = {c: n_samples / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    return np.array([class_weight[label] for label in y])


def load_data():
    """
    加载红酒质量数据集并进行特征工程。
    标签映射：quality 3-9 → 0-6（适配 XGBoost multi:softprob）

    Returns:
        X: 增强特征矩阵
        y: 映射后的标签数组（0-6）
        feature_names: 所有特征名称
        target_names: 原始质量等级名称（用于报告）
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"数据集不存在：{DATASET_PATH}\n请先运行 dataset/download_wine_data.py 下载数据"
        )

    df = pd.read_csv(DATASET_PATH)

    # 特征工程
    X, feature_names = feature_engineering(df)

    # 标签映射：quality 3-9 → 0-6
    y = df[TARGET_COLUMN].map(LABEL_MAP).values

    # 原始类别名称（用于输出报告）
    unique_orig = sorted(df[TARGET_COLUMN].unique())
    target_names = [str(c) for c in unique_orig]

    return X, y, feature_names, target_names


def train_model(
    X, y, test_size=0.2, random_state=42, n_iter_search=20, cv_folds=3
):
    """
    训练 XGBoost 分类模型（GPU 加速）并进行超参数调优。

    流程：
      1. 划分训练/测试集（分层抽样）
      2. 特征标准化
      3. 计算样本权重
      4. RandomizedSearchCV 搜索最优超参数（3折分层 CV）
      5. 使用早停重新训练最优模型
      6. 测试集评估

    Args:
        X: 特征数据
        y: 标签数据（已映射为 0-6）
        test_size: 测试集比例
        random_state: 随机种子
        n_iter_search: RandomizedSearch 迭代次数
        cv_folds: 交叉验证折数

    Returns:
        model: 训练好的最优模型
        scaler: 标准化器
        metrics: 评估指标字典（标签已映射回原始 quality）
        train_idx: 训练集索引
        test_idx: 测试集索引
        best_params: 最优超参数字典
    """
    # 划分训练集和测试集（分层抽样）
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, np.arange(len(y)),
        test_size=test_size, random_state=random_state, stratify=y,
    )

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 计算样本权重
    sample_weights = compute_sample_weights(y_train)

    # ── 超参数搜索空间（聚焦于经验证有效的高质量区域）──
    param_distributions = {
        "n_estimators": [200, 300, 400, 500, 600],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.08, 0.1, 0.12],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "min_child_weight": [3, 5, 7, 9],
        "gamma": [0, 0.05, 0.1, 0.2],
        "reg_alpha": [0, 0.01, 0.1, 0.5],
        "reg_lambda": [1.0, 2.0, 5.0, 10.0],
    }

    # XGBoost 基模型 —— 启用 GPU
    xgb_base = xgb.XGBClassifier(
        tree_method="hist",
        device="cuda",
        objective="multi:softprob",
        eval_metric=["mlogloss", "merror"],
        random_state=random_state,
        verbosity=0,
        n_jobs=4,
    )

    # 交叉验证策略
    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )

    print(f"\n  超参数搜索：{n_iter_search} 次随机采样, {cv_folds}-折 CV ({n_iter_search * cv_folds} 次 fit)")
    print(f"  使用 GPU: NVIDIA H20 (CUDA 12.8, 96GB)")

    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=n_iter_search,
        scoring="accuracy",
        cv=cv,
        verbose=0,
        n_jobs=1,                     # XGBoost 内部多线程，CV 串行避免 GPU 竞争
        random_state=random_state,
        error_score="raise",
    )

    search.fit(
        X_train_scaled, y_train,
        sample_weight=sample_weights,
    )

    # 获取最优参数
    best_model = search.best_estimator_
    best_params = search.best_params_

    print(f"\n  最优参数: {best_params}")
    print(f"  CV 最佳得分: {search.best_score_:.4f}")

    # ── 用全部训练数据重新训练最优模型（带早停）──
    # 从训练集中划分 10% 作为验证集用于早停
    X_tr, X_val, y_tr, y_val, sw_tr, _ = train_test_split(
        X_train_scaled, y_train, sample_weights,
        test_size=0.1,
        random_state=random_state,
        stratify=y_train,
    )

    # 最终模型使用更多树（早停会自动截断）
    final_n_estimators = max(best_params["n_estimators"] * 2, 800)

    final_model = xgb.XGBClassifier(
        **{k: v for k, v in best_params.items() if k != "n_estimators"},
        n_estimators=final_n_estimators,
        tree_method="hist",
        device="cuda",
        objective="multi:softprob",
        eval_metric=["mlogloss", "merror"],
        random_state=random_state,
        verbosity=0,
        early_stopping_rounds=50,
        n_jobs=4,
    )
    final_model.fit(
        X_tr, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # 获取实际训练的轮数
    actual_rounds = final_model.get_booster().num_boosted_rounds()
    print(f"  最终模型实际训练轮数: {actual_rounds} (设置 {final_n_estimators})")

    # ── 测试集评估（标签映射回原始 quality 值）──
    y_pred_mapped = final_model.predict(X_test_scaled)
    y_test_orig = np.array([INV_LABEL_MAP[v] for v in y_test])
    y_pred_orig = np.array([INV_LABEL_MAP[int(v)] for v in y_pred_mapped])

    accuracy = accuracy_score(y_test_orig, y_pred_orig)

    metrics = {
        "accuracy": accuracy,
        "y_test": y_test_orig,
        "y_pred": y_pred_orig,
        "y_test_mapped": y_test,
        "y_pred_mapped": y_pred_mapped,
        "X_test": X_test,
        "X_test_scaled": X_test_scaled,
        "X_train": X_train,
        "y_train": y_train,
    }

    return final_model, scaler, metrics, train_idx, test_idx, best_params


def save_model(model, scaler, best_params, save_path):
    """保存模型、标准化器和超参数到文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "best_params": best_params,
            "label_map": LABEL_MAP,
            "inv_label_map": INV_LABEL_MAP,
            "base_features": BASE_FEATURE_COLUMNS,
        },
        save_path,
    )
    print(f"模型已保存到：{save_path}")


def save_data_split(X_train, X_test, y_train, y_test, train_idx, test_idx, save_dir):
    """保存训练集和测试集的划分结果"""
    os.makedirs(save_dir, exist_ok=True)

    # 保存原始 quality 标签（已映射回 3-9）
    y_train_orig = np.array([INV_LABEL_MAP[v] for v in y_train])
    y_test_orig = np.array([INV_LABEL_MAP[v] for v in y_test])

    train_data = {
        "X": X_train,
        "y": y_train_orig,
        "indices": train_idx,
        "feature_columns": BASE_FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }
    train_path = os.path.join(save_dir, "train_data.joblib")
    joblib.dump(train_data, train_path)
    print(f"训练集已保存到：{train_path} (样本数：{len(y_train_orig)})")

    test_data = {
        "X": X_test,
        "y": y_test_orig,
        "indices": test_idx,
        "feature_columns": BASE_FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }
    test_path = os.path.join(save_dir, "test_data.joblib")
    joblib.dump(test_data, test_path)
    print(f"测试集已保存到：{test_path} (样本数：{len(y_test_orig)})")

    return train_path, test_path


def print_evaluation_report(y_test, y_pred, target_names):
    """打印详细评估报告（标签为原始 quality 值）"""
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n分类评估报告:")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("混淆矩阵:")
    print("=" * 60)
    unique_labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    header = "      " + "  ".join(f"{label:5s}" for label in [str(l) for l in unique_labels])
    print("预测值 →")
    print(header)
    print("真实值 ↓")
    for i, label in enumerate(unique_labels):
        print(f"  {label}: ", "  ".join(f"{v:5d}" for v in cm[i]))

    # 计算每个类别的召回率
    print("\n各类别召回率:")
    for i, label in enumerate(unique_labels):
        total = cm[i].sum()
        recall = cm[i, i] / total if total > 0 else 0.0
        print(f"  质量 {label}: {recall:.4f} ({cm[i, i]}/{total})")


def main():
    """主函数"""
    print("=" * 60)
    print("红酒质量预测任务 - XGBoost GPU 加速训练 v2.0")
    print("改进: wine_type + 特征工程 + 加权损失 + 超参搜索 + 早停")
    print("数据集：UCI Wine Quality (12 个基础特征 → 36 维增强特征)")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据与特征工程...")
    X, y, feature_names, target_names = load_data()
    print(f"基础特征数: {len(BASE_FEATURE_COLUMNS)}")
    print(f"对数变换特征: {len(LOG_TRANSFORM_COLS)}")
    print(f"酒类交互特征: 7")
    print(f"比例/组合特征: 9")
    print(f"平方项特征: 4")
    print(f"增强特征总数: {X.shape[1]}")
    print(f"特征列表: {feature_names}")
    print(f"样本数量: {len(y)}")
    print(f"类别数量: {len(target_names)} (质量等级: {', '.join(target_names)})")

    # 各类别样本分布（原始标签）
    print("\n各类别样本分布:")
    df_orig = pd.read_csv(DATASET_PATH)
    for label in target_names:
        count = np.sum(df_orig[TARGET_COLUMN] == int(label))
        print(f"  质量 {label}: {count:5d} 个样本 ({count/len(df_orig):.2%})")

    # 2. 特征标准化 + 超参数搜索 + 模型训练
    print("\n[2/5] 训练模型 (XGBoost GPU)...")
    print("模型：XGBoost (GPU: tree_method=hist, device=cuda)")

    model, scaler, metrics, train_idx, test_idx, best_params = train_model(
        X, y, test_size=0.2, random_state=42, n_iter_search=20, cv_folds=3,
    )
    print(f"\n测试集准确率：{metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"训练集样本数：{len(train_idx)}, 测试集样本数：{len(test_idx)}")

    # 3. 打印详细评估报告
    print("\n[3/5] 模型评估...")
    print_evaluation_report(metrics["y_test"], metrics["y_pred"], target_names)

    # 4. 保存模型和数据划分
    print("\n[4/5] 保存模型和数据集...")
    save_path = os.path.join(MODEL_DIR, "wine_model.joblib")
    save_model(model, scaler, best_params, save_path)

    save_data_split(
        metrics["X_train"],
        metrics["X_test"],
        metrics["y_train"],
        metrics["y_test_mapped"],
        train_idx,
        test_idx,
        MODEL_DIR,
    )

    # 5. 输出最优超参数
    print("\n[5/5] 输出最优超参数（供复现参考）:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

    return model, scaler, metrics


if __name__ == "__main__":
    main()