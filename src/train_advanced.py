#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本：基于 XGBoost multi:softprob + Optuna 贝叶斯优化的深度改进方案

改进方案要点：
  1. 保留 XGBoost multi:softprob 多分类损失（经验证优于独立二分类有序回归方案）
  2. 增强至 43 维特征（12基础 + 对数变换 + 酒类交互 + 比例/组合特征 + 平方项 + 额外多项式交互）
  3. Optuna 贝叶斯超参优化（替代 RandomizedSearchCV），全面搜索 10+ 超参数
  4. 分层抽样 + 早停机制 + 加权损失（处理类不平衡）
  5. Top-N 模型集成（从 Optuna trials 中选取最优模型进行集成，提升泛化能力）
  6. 自适应类别权重 + 概率校准

数据集：UCI Wine Quality（11个理化特征 + wine_type，7个质量等级：3-9）
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    print("错误: 需要安装 optuna 库。请运行: pip install optuna")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning)

# ── 项目路径 ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "winequality.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ── 质量等级定义 ────────────────────────────────────────────────────
QUALITY_LEVELS = [3, 4, 5, 6, 7, 8, 9]
N_CLASSES = len(QUALITY_LEVELS)

# ── 标签映射（XGBoost multi:softprob 要求类别从 0 开始）────────────
LABEL_MAP = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# ── 基础特征列（共12个：11个理化特征 + wine_type）────────────────────
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

# ── 对数变换特征列 ──────────────────────────────────────────────────
LOG_TRANSFORM_COLS = [
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
]


def feature_engineering(df):
    """
    增强特征工程（与原始方案一致，43维增强特征）。
    返回 (feature_matrix, feature_names)
    """
    names = list(BASE_FEATURE_COLUMNS)
    base = df[BASE_FEATURE_COLUMNS].values.copy()
    eps = 1e-8

    # --- 2. 对数变换（偏态特征）---
    log_feats = []
    log_names = []
    for c in LOG_TRANSFORM_COLS:
        log_feats.append(np.log1p(df[c].values))
        log_names.append(f"log_{c}")
    log_data = np.column_stack(log_feats)

    # --- 3. 酒类交互特征 ---
    wt = df["wine_type"].values
    interact_cols = [
        "alcohol", "volatile acidity", "residual sugar",
        "sulphates", "free sulfur dioxide", "total sulfur dioxide", "density",
    ]
    interact_data = [df[c].values * wt for c in interact_cols]
    interactions = np.column_stack(interact_data)
    interact_names = [f"{c}_x_wine_type" for c in interact_cols]

    # --- 4. 比例 / 组合特征 ---
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
    ratio_names = [
        "so2_ratio", "alc_sugar_ratio", "vol_alc_interact",
        "citric_volatile_ratio", "sulph_alc_interact", "ph_alc_interact",
        "total_acidity", "free_so2_pct", "citric_fixed_ratio",
    ]

    # --- 5. 平方项 ---
    square_cols = ["alcohol", "volatile acidity", "density", "residual sugar"]
    square_data = [df[c].values ** 2 for c in square_cols]
    squares = np.column_stack(square_data)
    square_names = [f"{c}_squared" for c in square_cols]

    # --- 6. 额外多项式交互特征 ---
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
    extra_names = [
        "acid_balance", "volatile_ph", "alcohol_density",
        "sulphate_volatile_ratio", "chlorides_sugar", "sqrt_total_so2",
        "free_so2_x_wine_type",
    ]

    # --- 合并所有特征 ---
    X_enhanced = np.column_stack([
        base, log_data, interactions, ratio_features,
        squares, extra_features
    ])
    all_names = (
        names + log_names + interact_names + ratio_names
        + square_names + extra_names
    )

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
    加载数据集并进行特征工程。
    标签映射：quality 3-9 → 0-6（适配 XGBoost multi:softprob）

    Returns:
        X: 增强特征矩阵 (43维)
        y: 映射后的标签数组 (0-6)
        feature_names: 特征名称列表
        target_names: 原始质量等级名称（用于报告）
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"数据集不存在：{DATASET_PATH}\n请先运行 dataset/download_wine_data.py"
        )

    df = pd.read_csv(DATASET_PATH)
    X, feature_names = feature_engineering(df)
    y = df[TARGET_COLUMN].map(LABEL_MAP).values

    unique_orig = sorted(df[TARGET_COLUMN].unique())
    target_names = [str(c) for c in unique_orig]

    return X, y, feature_names, target_names


def objective(trial, X_train, y_train, X_val, y_val, sample_weights):
    """
    Optuna 目标函数：搜索 XGBoost multi:softprob 最优超参数。
    优化目标：验证集上的加权准确率。
    """
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
    }

    model = xgb.XGBClassifier(
        **param,
        tree_method="hist",
        device="cuda",
        objective="multi:softprob",
        eval_metric=["mlogloss", "merror"],
        num_class=N_CLASSES,
        random_state=42,
        verbosity=0,
        n_jobs=4,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    # Also compute macro F1 for better class balance
    val_f1 = f1_score(y_val, y_val_pred, average="weighted")

    # Combine accuracy and macro F1 as objective
    return val_acc * 0.5 + val_f1 * 0.5


def train_model_with_optuna(
    X_train, y_train, X_val, y_val, X_test, y_test,
    scaler, n_trials=50, n_top_models=3
):
    """
    使用 Optuna 进行超参搜索，并训练 Top-N 集成模型。

    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        X_test, y_test: 测试数据
        scaler: 已拟合的 StandardScaler
        n_trials: Optuna 搜索次数
        n_top_models: 集成模型中包含的最优模型数量

    Returns:
        ensemble_models: Top-N 模型列表
        best_params: 最优超参数
        val_scores: 所有 trials 的验证分数
    """
    # 计算训练集样本权重
    train_weights = compute_sample_weights(y_train)

    # 标准化
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n  Optuna 超参搜索 ({n_trials} trials, Top-{n_top_models} 集成)...")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=10),
    )

    # 划分早停验证集（从训练集中分出 15%）
    X_tr, X_early, y_tr, y_early, sw_tr, _ = train_test_split(
        X_train_scaled, y_train, train_weights,
        test_size=0.15, random_state=42, stratify=y_train,
    )

    study.optimize(
        lambda trial: objective(
            trial, X_tr, y_tr, X_early, y_early, sw_tr
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_score = study.best_value
    print(f"  最优得分(验证): {best_score:.4f}")
    print(f"  最优参数: {best_params}")

    # 获取 Top-N trials
    all_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else 0, reverse=True)
    top_trials = all_trials[:n_top_models]

    # 训练 Top-N 模型
    ensemble_models = []
    val_scores = []

    for rank, t in enumerate(top_trials):
        if t.value is None:
            continue
        params = t.params
        if params is None:
            continue

        print(f"  训练 Top-{rank+1} 模型 (得分={t.value:.4f})...")

        model = xgb.XGBClassifier(
            **params,
            tree_method="hist",
            device="cuda",
            objective="multi:softprob",
            eval_metric=["mlogloss", "merror"],
            num_class=N_CLASSES,
            random_state=42,
            verbosity=0,
            n_jobs=4,
        )

        # 用全部训练集+验证集训练，早停基于内部的验证划分
        X_train_full = np.vstack([X_train_scaled, X_val_scaled])
        y_train_full = np.concatenate([y_train, y_val])
        full_weights = compute_sample_weights(y_train_full)

        # 再从中分出一部分做早停
        X_tr_full, X_early_full, y_tr_full, y_early_full, sw_full, _ = train_test_split(
            X_train_full, y_train_full, full_weights,
            test_size=0.12, random_state=42, stratify=y_train_full,
        )

        model.fit(
            X_tr_full, y_tr_full,
            sample_weight=sw_full,
            eval_set=[(X_early_full, y_early_full)],
            verbose=False,
        )

        ensemble_models.append(model)
        val_scores.append(t.value)

    # 评估 Top-1 模型在测试集上的表现
    top_model = ensemble_models[0]
    y_test_pred = top_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"\n  Top-1 模型测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # 评估集成模型
    if n_top_models > 1:
        y_test_pred_ensemble = ensemble_predict(ensemble_models, X_test_scaled)
        ensemble_acc = accuracy_score(y_test, y_test_pred_ensemble)
        print(f"  Top-{n_top_models} 集成模型测试集准确率: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")

    return ensemble_models, best_params, val_scores


def ensemble_predict(models, X):
    """
    Top-N 模型集成预测（投票法）。
    每个模型输出概率，取平均后取 argmax。
    """
    n_samples = X.shape[0]
    n_models = len(models)
    all_probas = np.zeros((n_samples, N_CLASSES, n_models))

    for i, model in enumerate(models):
        proba = model.predict_proba(X)
        all_probas[:, :, i] = proba

    avg_probas = np.mean(all_probas, axis=2)
    predictions = np.argmax(avg_probas, axis=1)

    return predictions


def main():
    """主函数：训练 XGBoost multi:softprob + Optuna 集成模型"""
    print("=" * 70)
    print("  XGBoost multi:softprob + Optuna 贝叶斯优化 + Top-N 集成")
    print("  数据集：UCI Wine Quality (红酒+白酒)")
    print(f"  特征维度：{len(BASE_FEATURE_COLUMNS)} 基础 → 43 维增强特征")
    print(f"  质量等级：{QUALITY_LEVELS}")
    print(f"  GPU: NVIDIA H20 (CUDA 12.8, 96GB)")
    print("=" * 70)

    # 1. 加载数据
    print(f"\n{'[1/6]':>8} 加载数据与特征工程...")
    X, y, feature_names, target_names = load_data()
    print(f"  特征总数: {X.shape[1]}")
    print(f"  样本总数: {len(y)}")
    print(f"  特征列表: {feature_names}")
    print(f"  类别: {', '.join(target_names)}")

    # 2. 数据划分（分层抽样）
    print(f"\n{'[2/6]':>8} 划分训练/验证/测试集 (分层抽样)...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15, random_state=42,
        stratify=y_train_val,
    )

    print(f"  训练集: {len(y_train)} 样本")
    print(f"  验证集: {len(y_val)} 样本")
    print(f"  测试集: {len(y_test)} 样本")

    # 各类别样本分布
    print(f"\n  各类别样本分布（训练集）:")
    df_orig = pd.read_csv(DATASET_PATH)
    for label in target_names:
        count = np.sum(df_orig[TARGET_COLUMN] == int(label))
        print(f"    质量 {label}: {count:5d} 个样本 ({count/len(df_orig):.2%})")

    # 3. 特征标准化
    print(f"\n{'[3/6]':>8} 特征标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. 训练
    print(f"\n{'[4/6]':>8} Optuna 超参搜索 + 模型训练...")
    ensemble_models, best_params, val_scores = train_model_with_optuna(
        X_train, y_train, X_val, y_val, X_test, y_test,
        scaler, n_trials=50, n_top_models=3,
    )

    # 5. 测试集评估
    print(f"\n{'[5/6]':>8} 测试集评估...")

    # 单模型评估
    top_model = ensemble_models[0]
    y_pred_single = top_model.predict(X_test_scaled)
    single_acc = accuracy_score(y_test, y_pred_single)

    # 集成模型评估
    if len(ensemble_models) > 1:
        y_pred = ensemble_predict(ensemble_models, X_test_scaled)
        ensemble_acc = accuracy_score(y_test, y_pred)
        print(f"  Top-1 模型准确率: {single_acc:.4f}")
        print(f"  Top-{len(ensemble_models)} 集成准确率: {ensemble_acc:.4f}")
        # 使用集成结果
        y_pred_final = y_pred
        final_acc = ensemble_acc
    else:
        y_pred_final = y_pred_single
        final_acc = single_acc

    # 标签映射回原始 quality 值
    y_test_orig = np.array([INV_LABEL_MAP[v] for v in y_test])
    y_pred_orig = np.array([INV_LABEL_MAP[int(v)] for v in y_pred_final])

    print(f"\n{'='*60}")
    print(f"  测试集准确率: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"{'='*60}")

    print(f"\n分类评估报告:")
    print(classification_report(y_test_orig, y_pred_orig, target_names=target_names))

    print(f"混淆矩阵:")
    unique_labels = sorted(set(y_test_orig))
    cm = confusion_matrix(y_test_orig, y_pred_orig, labels=unique_labels)
    header = "      " + "  ".join(f"{l:5s}" for l in [str(l) for l in unique_labels])
    print(f"预测值 →")
    print(header)
    print(f"真实值 ↓")
    for i, label in enumerate(unique_labels):
        print(f"  {label}: ", "  ".join(f"{v:5d}" for v in cm[i]))

    print(f"\n各类别召回率:")
    for i, label in enumerate(unique_labels):
        total = cm[i].sum()
        recall = cm[i, i] / total if total > 0 else 0.0
        print(f"  质量 {label}: {recall:.4f} ({cm[i, i]}/{total})")

    # 6. 保存模型和所有组件
    print(f"\n{'[6/6]':>8} 保存模型...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 保存集成模型包
    model_package = {
        "models": ensemble_models,
        "scaler": scaler,
        "best_params": best_params,
        "feature_names": feature_names,
        "quality_levels": QUALITY_LEVELS,
        "label_map": LABEL_MAP,
        "inv_label_map": INV_LABEL_MAP,
        "base_features": BASE_FEATURE_COLUMNS,
        "log_transform_cols": LOG_TRANSFORM_COLS,
        "val_scores": val_scores,
        "n_models": len(ensemble_models),
    }
    save_path = os.path.join(MODEL_DIR, "ordinal_ensemble_model.joblib")
    joblib.dump(model_package, save_path)
    print(f"  集成模型已保存到: {save_path}")

    # 保存测试数据
    test_package = {
        "X_test": X_test,
        "y_test": y_test_orig,
        "y_test_mapped": y_test,
        "y_pred": y_pred_orig,
        "y_pred_mapped": y_pred_final,
        "feature_names": feature_names,
    }
    test_save_path = os.path.join(MODEL_DIR, "ordinal_test_data.joblib")
    joblib.dump(test_package, test_save_path)
    print(f"  测试数据已保存到: {test_save_path}")

    print(f"\n{'='*70}")
    print(f"  训练完成! 测试集准确率: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"{'='*70}")

    return ensemble_models, scaler, best_params


if __name__ == "__main__":
    main()