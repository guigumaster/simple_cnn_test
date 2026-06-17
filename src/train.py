#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本：基于 GPU 加速 XGBoost 的多策略融合方案
改进点：
  1. GPU 加速 XGBoost 替代 RandomForest
  2. 补全缺失特征 'wine_type'（红酒/白酒判别信息）
  3. 自定义类别权重 + SMOTE 过采样处理类别不平衡
  4. 交互特征工程（多项式交互特征）
数据集：UCI Wine Quality（合并红酒 + 白酒，含 wine_type 特征）
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, recall_score, precision_score
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb


# ========== 路径配置 ==========
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'winequality.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# ========== 特征配置 ==========
# 原始 11 个理化特征 + wine_type（下载时已添加，但原训练代码未使用）
BASE_FEATURE_COLUMNS = [
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

# 关键新增：wine_type 特征（0=红酒, 1=白酒）
EXTRA_FEATURES = ['wine_type']

# 完整特征列表（训练时使用）
ALL_FEATURES = BASE_FEATURE_COLUMNS + EXTRA_FEATURES

TARGET_COLUMN = 'quality'
# ========== 质量标签映射（原始 [3,4,5,6,7,8,9] → 0-indexed [0,1,2,3,4,5,6]）==========
# XGBoost 要求类别标签从 0 开始连续编号
QUALITY_MAP = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}
QUALITY_INV_MAP = {v: k for k, v in QUALITY_MAP.items()}  # 逆映射，用于预测还原
QUALITY_LABELS = ['3', '4', '5', '6', '7', '8', '9']

# ========== XGBoost 超参数 ==========
XGB_PARAMS = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    # 'num_class' 将在 train_model 中根据实际类别数动态设置
    'learning_rate': 0.03,
    'n_estimators': 800,
    'max_depth': 5,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 3,
    'reg_lambda': 2.0,
    'reg_alpha': 1.0,
    'gamma': 0.2,
    'tree_method': 'hist',
    'device': 'cuda',
    'n_jobs': -1,
    'random_state': 42,
}

# ========== 交互特征配置 ==========
INTERACTION_DEGREE = 2        # 二阶交互
INTERACTION_ONLY = True       # 只保留交互项（不含平方项）

# ========== 是否使用交互特征 ==========
# ★ 关闭交互特征以降低过拟合风险（78维→12维），提升泛化能力
USE_INTERACTION_FEATURES = True


def load_data():
    """
    加载红酒+白酒合并数据集，包含 wine_type 特征

    Returns:
        df: 完整 DataFrame
        feature_names: 全部特征名（包含 wine_type）
        target_names: 类别名称
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"数据集不存在：{DATASET_PATH}\n"
            f"请先运行：python {os.path.join(PROJECT_ROOT, 'dataset', 'download_wine_data.py')}"
        )

    df = pd.read_csv(DATASET_PATH)

    # 确保 wine_type 列存在
    if 'wine_type' not in df.columns:
        print("[警告] 数据集中未找到 'wine_type' 列，将根据特征差异推断...")
        # 白酒的总二氧化硫通常显著高于红酒，作为 fallback 推断
        median_tsd = df['total sulfur dioxide'].median()
        df['wine_type'] = (df['total sulfur dioxide'] > median_tsd).astype(int)

    # 检查缺失值
    missing = df[ALL_FEATURES].isnull().sum()
    if missing.sum() > 0:
        print(f"[信息] 发现缺失值，使用中位数填充：\n{missing[missing > 0]}")
        df[ALL_FEATURES] = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median())

    # 类别标签（原始值，未映射）
    unique_classes = sorted(df[TARGET_COLUMN].unique())
    target_names = [str(c) for c in unique_classes]

    print(f"数据形状：{df.shape}")
    print(f"样本数量：{len(df)}")
    print(f"特征数量（含 wine_type）：{len(ALL_FEATURES)}")
    print(f"类别数量：{len(unique_classes)} (质量等级：{', '.join(target_names)})")

    return df, ALL_FEATURES, target_names


def create_interaction_features(X_df, base_feature_names):
    """
    构建二阶交互特征（PolynomialFeatures, interaction_only=True）

    Args:
        X_df: 原始特征 DataFrame
        base_feature_names: 基特征名称列表

    Returns:
        X_interact_df: 包含原始特征 + 交互特征的 DataFrame
        interaction_feature_names: 所有特征名称
        poly: 训练好的 PolynomialFeatures 对象
    """
    poly = PolynomialFeatures(
        degree=INTERACTION_DEGREE,
        interaction_only=INTERACTION_ONLY,
        include_bias=False
    )

    X_poly = poly.fit_transform(X_df[base_feature_names])
    poly_feature_names = poly.get_feature_names_out(base_feature_names)

    # 构建 DataFrame
    X_interact_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X_df.index)

    print(f"  基特征数量：{len(base_feature_names)}")
    print(f"  交互特征数量：{len(poly_feature_names)}")
    print(f"  新增交互特征示例：{poly_feature_names[len(base_feature_names):][:5]}")

    return X_interact_df, poly_feature_names, poly


def build_sample_weights(y, class_weight_dict=None):
    """
    基于类别频率计算样本权重（用于 XGBoost sample_weight）

    Args:
        y: 标签数组
        class_weight_dict: 自定义权重字典，None 则自动计算

    Returns:
        sample_weights: 每个样本对应的权重
        weight_dict: 类别 -> 权重 映射
    """
    if class_weight_dict is None:
        # 使用 balanced 策略：n_samples / (n_classes * np.bincount(y))
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))

    sample_weights = np.array([class_weight_dict[label] for label in y])
    return sample_weights, class_weight_dict


def train_model(df, test_size=0.2, random_state=42):
    """
    训练 XGBoost 模型（GPU 加速），包含完整的多策略融合流程

    Args:
        df: 完整 DataFrame
        test_size: 测试集比例
        random_state: 随机种子

    Returns:
        model: 训练好的 XGBoost 模型
        scaler: 标准化器
        poly: 多项式交互特征生成器
        metrics: 评估指标字典
        train_idx: 训练集索引
        test_idx: 测试集索引
        class_weight_dict: 类别权重字典
    """
    # ---------------------------------------------------------------
    # Step 1: 划分训练集 / 测试集（分层抽样）
    # ---------------------------------------------------------------
    X_base = df[ALL_FEATURES].values
    y_orig = df[TARGET_COLUMN].values  # 原始标签 (3-9)

    # 将质量标签映射为 0-indexed: [3,4,5,6,7,8,9] → [0,1,2,3,4,5,6]
    y = np.array([QUALITY_MAP[val] for val in y_orig])

    X_train_base, X_test_base, y_train, y_test, train_idx, test_idx = train_test_split(
        X_base, y, np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 保存原始标签（用于报告输出）
    y_test_orig = y_orig[test_idx]

    # 转为 DataFrame 便于特征工程
    X_train_df = pd.DataFrame(X_train_base, columns=ALL_FEATURES)
    X_test_df = pd.DataFrame(X_test_base, columns=ALL_FEATURES)

    # ---------------------------------------------------------------
    # Step 2: 交互特征工程（可选）
    # ---------------------------------------------------------------
    print("\n[特征工程]")
    print("-" * 40)
    if USE_INTERACTION_FEATURES:
        print("基特征（含 wine_type）：", ALL_FEATURES)
        X_train_poly, poly_feature_names, poly = create_interaction_features(
            X_train_df, ALL_FEATURES
        )
        X_test_poly = pd.DataFrame(
            poly.transform(X_test_df),
            columns=poly_feature_names,
            index=X_test_df.index
        )
        print(f"\n训练集特征形状（含交互）：{X_train_poly.shape}")
        print(f"测试集特征形状（含交互）：{X_test_poly.shape}")
    else:
        poly = None
        print("使用原始基特征（不含交互特征），特征数：", len(ALL_FEATURES))
        X_train_poly = X_train_df
        X_test_poly = X_test_df

    # ---------------------------------------------------------------
    # Step 3: 特征标准化 + SMOTE 过采样
    # ---------------------------------------------------------------
    print("\n[特征标准化 + SMOTE 过采样]")
    print("-" * 40)

    # 先标准化，再 SMOTE（SMOTE 要求数值特征）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    # ★ 保存 poly 是否为 None，供后续步骤使用
    if poly is not None:
        all_poly_names = poly.get_feature_names_out(ALL_FEATURES)
    else:
        all_poly_names = ALL_FEATURES.copy()

    # 打印原始类别分布
    print("\nSMOTE 之前的训练集类别分布（映射后）:")
    train_class_counts = pd.Series(y_train).value_counts().sort_index()
    for cls, cnt in train_class_counts.items():
        orig_cls = QUALITY_INV_MAP[cls]
        print(f"  质量 {orig_cls} (映射为 {cls}): {cnt} 个样本")

    # ---------------------------------------------------------------
    # SMOTE 过采样（平衡各类别）
    # ★ 只使用 SMOTE，不使用 sample_weight，避免双重加权
    # ---------------------------------------------------------------
    smote_k = min(3, np.min(pd.Series(y_train).value_counts()) - 1)
    smote_k = max(1, smote_k)
    smote = SMOTE(
        sampling_strategy='auto',
        k_neighbors=smote_k,
        random_state=random_state
    )
    try:
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    except ValueError as e:
        print(f"  [警告] SMOTE 失败 ({e})，回退至 RandomOverSampler")
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=random_state)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)

    print("\nSMOTE 之后的训练集类别分布:")
    resampled_counts = pd.Series(y_train_resampled).value_counts().sort_index()
    for cls, cnt in resampled_counts.items():
        orig_cls = QUALITY_INV_MAP[cls]
        print(f"  质量 {orig_cls} (映射为 {cls}): {cnt} 个样本")
    print(f"\n训练集样本数: {len(y_train)} -> {len(y_train_resampled)} (增补 {len(y_train_resampled) - len(y_train)})")

    # ---------------------------------------------------------------
    # Step 4: 自定义类别权重
    # ---------------------------------------------------------------
    print("\n[自定义类别权重]")
    print("-" * 40)

    # ★ SMOTE 已平衡所有类别，不使用额外 sample_weight

    # ---------------------------------------------------------------
    # Step 5: 训练 GPU 加速 XGBoost
    # ---------------------------------------------------------------
    print("\n[训练 GPU 加速 XGBoost]")
    print("-" * 40)
    print(f"XGBoost 参数:")
    for k, v in XGB_PARAMS.items():
        print(f"  {k}: {v}")

    # 动态设置 num_class
    params = dict(XGB_PARAMS)
    n_classes = len(np.unique(y))
    params['num_class'] = n_classes

    # --- 第一阶段：带 early_stopping 的训练（确定最佳迭代轮数）---
    early_stop_params = dict(params)
    early_stop_params['early_stopping_rounds'] = 50
    model_es = xgb.XGBClassifier(**early_stop_params)
    model_es.fit(
        X_train_resampled, y_train_resampled,
        eval_set=[(X_train_resampled, y_train_resampled), (X_test_scaled, y_test)],
        verbose=False
    )

    # --- 第二阶段：用最佳轮数重新拟合整个训练集（提升泛化能力）---
    if hasattr(model_es, 'best_iteration') and model_es.best_iteration is not None:
        best_iter = int(model_es.best_iteration)
        print(f"  提前停止：最佳迭代轮数 = {best_iter}")
        best_params = dict(params)
        best_params['n_estimators'] = best_iter
        model = xgb.XGBClassifier(**best_params)
        model.fit(
            X_train_resampled, y_train_resampled,
            verbose=False
        )
    else:
        model = model_es

    # 用最佳迭代次数重新拟合（防止过拟合）
    if hasattr(model, 'best_iteration') and model.best_iteration is not None:
        best_iter = int(model.best_iteration)
        print(f"  提前停止：最佳迭代轮数 = {best_iter}")
        # 重新用最佳轮次训练
        best_params = dict(params)
        best_params['n_estimators'] = best_iter
        model = xgb.XGBClassifier(**best_params)
        model.fit(
            X_train_resampled, y_train_resampled,
            verbose=False
        )

    # ---------------------------------------------------------------
    # Step 6: 评估
    # ---------------------------------------------------------------
    print("\n[模型评估]")
    print("-" * 40)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    # ★ 将预测结果映射回原始标签
    y_pred_orig = np.array([QUALITY_INV_MAP[p] for p in y_pred])
    y_test_orig_display = np.array([QUALITY_INV_MAP[t] for t in y_test])

    metrics = {
        'accuracy': accuracy,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_test_orig': y_test_orig_display,   # 原始标签（用于报告）
        'y_pred_orig': y_pred_orig,            # 映射回原始标签
        'X_test_original': X_test_base,
        'X_test_scaled': X_test_scaled,
        'X_train_original': X_train_base,
        'y_train': y_train,
        'y_train_resampled': y_train_resampled,
    }

    return model, scaler, poly, metrics, train_idx, test_idx, {}


def print_evaluation_report(y_test, y_pred, target_names):
    """打印详细评估报告"""
    print("\n" + "=" * 70)
    print("分 类 评 估 报 告")
    print("=" * 70)

    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print(report)

    # 额外计算宏平均和加权平均指标
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    macro_recall = recall_score(y_test, y_pred, average='macro')
    macro_precision = precision_score(y_test, y_pred, average='macro')

    print("\n综合指标:")
    print(f"  准确率 (Accuracy):      {accuracy_score(y_test, y_pred):.4f} ({accuracy_score(y_test, y_pred)*100:.2f}%)")
    print(f"  宏平均精确率 (Macro P): {macro_precision:.4f}")
    print(f"  宏平均召回率 (Macro R): {macro_recall:.4f}")
    print(f"  宏平均 F1 (Macro F1):  {macro_f1:.4f}")
    print(f"  加权平均 F1 (Weighted): {weighted_f1:.4f}")

    print("\n混淆矩阵:")
    print("=" * 70)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
    unique_labels = sorted(set(y_test))
    # 打印列标题
    header = "         " + "  ".join(f"{l:>5}" for l in unique_labels)
    print("预测值:")
    print(header)
    print("真实值:")
    for i, label in enumerate(unique_labels):
        row = f"  {label:>2}:    " + "  ".join(f"{v:>5}" for v in cm[i])
        print(row)

    # 每类详细指标
    print("\n各类别详细指标:")
    print("  " + "-" * 55)
    print(f"  {'质量等级':>6} | {'精确率':>8} | {'召回率':>8} | {'F1分数':>8} | {'样本数':>6}")
    print("  " + "-" * 55)
    from sklearn.metrics import precision_recall_fscore_support
    per_class = precision_recall_fscore_support(y_test, y_pred, labels=sorted(set(y_test)))
    for i, label in enumerate(sorted(set(y_test))):
        print(f"  {label:>6} | {per_class[0][i]:>8.4f} | {per_class[1][i]:>8.4f} | {per_class[2][i]:>8.4f} | {per_class[3][i]:>6}")
    print("  " + "-" * 55)


def save_all(model, scaler, poly, class_weight_dict, metrics, train_idx, test_idx):
    """
    保存模型、特征工程器、数据划分和配置

    Args:
        model: XGBoost 模型
        scaler: 标准化器
        poly: PolynomialFeatures 对象
        class_weight_dict: 类别权重字典
        metrics: 评估指标
        train_idx: 训练集索引
        test_idx: 测试集索引
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. 保存模型 + 所有预处理组件
    artifact_path = os.path.join(MODEL_DIR, 'wine_model.joblib')
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'poly': poly,  # 若 USE_INTERACTION_FEATURES=False 则为 None
        'use_interaction': USE_INTERACTION_FEATURES,
        'feature_columns': ALL_FEATURES,
        'class_weight_dict': class_weight_dict,
        'model_params': dict(XGB_PARAMS),
        'quality_map': QUALITY_MAP,
        'quality_inv_map': QUALITY_INV_MAP,
        'random_state': 42,
    }, artifact_path)
    print(f"\n模型及预处理组件已保存到：{artifact_path}")

    # 2. 保存训练集
    train_data = {
        'X': metrics['X_train_original'],
        'y': metrics['y_train'],
        'y_resampled': metrics['y_train_resampled'],
        'indices': train_idx,
        'feature_columns': ALL_FEATURES,
        'target_column': TARGET_COLUMN,
    }
    train_path = os.path.join(MODEL_DIR, 'train_data.joblib')
    joblib.dump(train_data, train_path)
    print(f"训练集已保存到：{train_path} (样本数：{len(metrics['y_train'])})")

    # 3. 保存测试集
    test_data = {
        'X': metrics['X_test_original'],
        'y': metrics['y_test'],
        'indices': test_idx,
        'feature_columns': ALL_FEATURES,
        'target_column': TARGET_COLUMN,
    }
    test_path = os.path.join(MODEL_DIR, 'test_data.joblib')
    joblib.dump(test_data, test_path)
    print(f"测试集已保存到：{test_path} (样本数：{len(metrics['y_test'])})")


def main():
    """主函数"""
    print("=" * 70)
    print("🍷 红酒质量预测 - GPU 加速 XGBoost 多策略融合训练")
    print("   改进方案：GPU XGBoost + wine_type + SMOTE + 类别权重 + 交互特征")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    df, feature_names, target_names = load_data()

    print("\n各类别样本分布（全数据集）:")
    for label in sorted(df[TARGET_COLUMN].unique()):
        count = (df[TARGET_COLUMN] == label).sum()
        pct = count / len(df) * 100
        print(f"  质量 {label}: {count} 个样本 ({pct:.2f}%)")

    # wine_type 分布
    wine_type_counts = df['wine_type'].value_counts().sort_index()
    print(f"\n酒类分布:")
    print(f"  红酒 (wine_type=0): {wine_type_counts.get(0, 0)} 个样本")
    print(f"  白酒 (wine_type=1): {wine_type_counts.get(1, 0)} 个样本")

    # 2. 训练模型
    print("\n[2/5] 训练模型（GPU 加速 XGBoost）...")
    model, scaler, poly, metrics, train_idx, test_idx, class_weight_dict = train_model(df)
    print(f"\n>> 测试集准确率：{metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f">> 训练集样本数：{len(train_idx)}, 测试集样本数：{len(test_idx)}")

    # 3. 打印评估报告
    print("\n[3/5] 模型评估...")
    print_evaluation_report(metrics['y_test_orig'], metrics['y_pred_orig'], target_names)

    # 4. 保存模型和数据
    print("\n[4/5] 保存模型、预处理组件和数据划分...")
    save_all(model, scaler, poly, class_weight_dict, metrics, train_idx, test_idx)

    # 5. 输出特征重要性
    print("\n[5/5] 特征重要性分析（Top 20）...")
    print("=" * 70)
    if hasattr(model, 'feature_importances_'):
        # 获取特征名称（含交互或不含交互）
        if USE_INTERACTION_FEATURES:
            all_poly_names = poly.get_feature_names_out(ALL_FEATURES)
        else:
            all_poly_names = ALL_FEATURES
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        print(f"{'排名':>4} | {'特征名':<35} | {'重要性':>10}")
        print("-" * 55)
        for rank, idx in enumerate(sorted_idx[:20], 1):
            name = all_poly_names[idx]
            imp = importances[idx]
            print(f"{rank:>4} | {name:<35} | {imp:>10.6f}")

    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"模型保存位置：{os.path.join(MODEL_DIR, 'wine_model.joblib')}")
    print("=" * 70)

    return model, scaler, poly, metrics


if __name__ == "__main__":
    main()