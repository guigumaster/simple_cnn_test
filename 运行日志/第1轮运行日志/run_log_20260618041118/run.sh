#!/bin/bash
# ============================================================
# 红酒质量预测 —— 有序回归 + SMOTE-ENN + Optuna 深度改进方案
# 标题：基于有序回归(Ordinal Regression) + SMOTE-ENN混合采样 +
#        Optuna贝叶斯优化的XGBoost深度改进方案
# 描述：将当前忽略标签序数关系的多分类损失函数替换为有序回归损失，
#       通过训练6个二分类器（预测quality≥4至≥9）利用质量等级间的序数关系；
#       结合SMOTE-ENN对稀有类别过采样并清洗噪声；
#       使用Optuna进行高效贝叶斯超参优化；
#       引入自适应阈值校准和分层集成策略，
#       系统性提升模型对不平衡、有序多分类任务的性能。
# ============================================================
set -euo pipefail

# ── 项目根目录（绝对路径）───────────────────────────────────────────
PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/514bde8e-62f3-47f4-b193-f8785ddf8e2b/simple_cnn_test/code/4d2cc467-1e2d-4692-b34d-fa25eb619a9a/simple_cnn_test"

# 目录定义
RUN_LOG_DIR="${PROJECT_ROOT}/run_log"
DATASET_DIR="${PROJECT_ROOT}/dataset"
MODEL_DIR="${PROJECT_ROOT}/models"
SRC_DIR="${PROJECT_ROOT}/src"

# 时间戳 & 日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${RUN_LOG_DIR}/run_advanced_${TIMESTAMP}.log"

# 确保目录存在
mkdir -p "${RUN_LOG_DIR}" "${MODEL_DIR}"

# ── 日志输出 —— 同时输出到控制台和日志文件 ────────────────────────
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo " 红酒质量预测 —— 有序回归 + SMOTE-ENN + Optuna 深度改进方案"
echo " 项目根目录 : ${PROJECT_ROOT}"
echo " 日志文件   : ${LOG_FILE}"
echo " 时间戳     : ${TIMESTAMP}"
echo "============================================================"
echo ""

# ── 0. 检查硬件环境 ──────────────────────────────────────────────
echo ">>> [0/7] 检查硬件环境..."
echo "  CPU 信息: $(grep 'model name' /proc/cpuinfo | head -1)"
echo "  CPU 核心数: $(nproc)"
echo "  内存信息: $(free -h | grep Mem | awk '{print $2}')"
echo ""

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || nvidia-smi
fi
echo ""

# ── 1. 数据集准备 ────────────────────────────────────────────────
echo ">>> [1/7] 检查/下载数据集..."
if [ -f "${DATASET_DIR}/winequality.csv" ]; then
    echo "  数据集已存在: ${DATASET_DIR}/winequality.csv"
    echo "  样本数: $(wc -l < "${DATASET_DIR}/winequality.csv") (含表头)"
else
    echo "  下载数据集..."
    python3 "${DATASET_DIR}/download_wine_data.py"
fi
echo ""

# ── 2. 环境检查 ──────────────────────────────────────────────────
echo ">>> [2/7] 检查 Python 环境..."
echo "  Python: $(python3 --version)"
echo "  pip: $(pip3 --version | awk '{print $2}')"

# 检查关键依赖
python3 -c "import xgboost; print(f'  XGBoost: {xgboost.__version__}')"
python3 -c "import sklearn; print(f'  scikit-learn: {sklearn.__version__}')"
python3 -c "import pandas; print(f'  pandas: {pandas.__version__}')"
python3 -c "import numpy; print(f'  numpy: {numpy.__version__}')"
python3 -c "import joblib; print(f'  joblib: {joblib.__version__}')"
python3 -c "import imblearn; print(f'  imbalanced-learn: {imblearn.__version__}')"
echo ""

# 安装 Optuna（如果未安装）
echo ">>> [2b/7] 确保 Optuna 已安装..."
python3 -c "import optuna" 2>/dev/null && \
    echo "  Optuna: $(python3 -c 'import optuna; print(optuna.__version__)')" || \
    (echo "  正在安装 Optuna..." && pip3 install optuna)
echo ""

# ── 3. 数据分布分析 ──────────────────────────────────────────────
echo ">>> [3/7] 数据分布分析..."
python3 -c "
import pandas as pd
df = pd.read_csv('${DATASET_DIR}/winequality.csv')
print(f'总样本数: {len(df)}')
print(f'质量等级分布:')
for v, c in df['quality'].value_counts().sort_index().items():
    print(f'  quality {v}: {c:5d} ({c/len(df)*100:.2f}%)')
print(f'红酒: {(df[\"wine_type\"]==0).sum()}, 白酒: {(df[\"wine_type\"]==1).sum()}')
print()
print('有序回归阈值分布:')
for k in [4,5,6,7,8,9]:
    pos = (df['quality'] >= k).sum()
    neg = (df['quality'] < k).sum()
    print(f'  >= {k}: 正样本={pos:5d}, 负样本={neg:5d}, 比例={pos/neg:.3f}')
"
echo ""

# ── 4. 有序回归集成模型训练（GPU 加速 + SMOTE-ENN + Optuna） ─────
echo ">>> [4/7] 开始训练有序回归集成模型..."
echo "  训练脚本: ${SRC_DIR}/train_advanced.py"
echo "  开始时间: $(date)"
echo "  训练策略:"
echo "    1. 6个二分类器 (quality≥4至≥9) 利用序数关系"
echo "    2. SMOTE-ENN 混合采样处理类别不平衡"
echo "    3. Optuna 贝叶斯超参优化 (每个二分类器独立搜索)"
echo "    4. 自适应阈值校准"
echo "    5. 分层集成策略"
echo ""

time python3 "${SRC_DIR}/train_advanced.py"

echo ""
echo "  结束时间: $(date)"
echo "  训练完成!"
echo ""

# ── 5. 有序回归集成模型预测 ──────────────────────────────────────
echo ">>> [5/7] 有序回归集成模型预测..."
echo "  预测脚本: ${SRC_DIR}/predict_advanced.py"
echo "  开始时间: $(date)"
echo ""

time python3 "${SRC_DIR}/predict_advanced.py"

echo ""
echo "  结束时间: $(date)"
echo ""

# ── 6. 与原始模型对比评估 ────────────────────────────────────────
echo ">>> [6/7] 与原始 XGBoost 模型对比评估..."
echo "  开始时间: $(date)"
echo ""

# 检查原始模型是否存在
ORIGINAL_MODEL="${MODEL_DIR}/wine_model.joblib"
if [ -f "${ORIGINAL_MODEL}" ]; then
    echo "  原始模型存在, 运行原始预测进行对比..."
    echo "  ---------- 原始模型预测 (XGBoost GPU baseline) ----------"
    time python3 "${SRC_DIR}/predict.py" 2>&1 | tail -30 || echo "  原始预测脚本运行异常(可能因版本差异)"
else
    echo "  原始模型不存在, 跳过对比"
fi
echo ""

echo "  结束时间: $(date)"
echo ""

# ── 7. 输出结果摘要 ──────────────────────────────────────────────
echo ">>> [7/7] 输出结果摘要..."
echo ""

# 有序回归集成模型信息
ENSEMBLE_MODEL="${MODEL_DIR}/ordinal_ensemble_model.joblib"
if [ -f "${ENSEMBLE_MODEL}" ]; then
    MODEL_SIZE=$(du -h "${ENSEMBLE_MODEL}" | cut -f1)
    echo "  有序回归集成模型: ${ENSEMBLE_MODEL} (${MODEL_SIZE})"
    echo "  包含 6 个 XGBoost 二分类器 + 自适应阈值 + 标准化器"
fi

TEST_DATA="${MODEL_DIR}/ordinal_test_data.joblib"
if [ -f "${TEST_DATA}" ]; then
    echo "  测试数据文件: ${TEST_DATA}"
fi

echo "  日志文件: ${LOG_FILE}"
echo ""

# 从日志中提取准确率
if [ -f "${LOG_FILE}" ]; then
    echo "  ---------- 性能摘要 ----------"
    grep -E "测试集准确率|有序回归集成模型" "${LOG_FILE}" | tail -5
    echo ""
    echo "  ---------- 各类别召回率 ----------"
    grep -A 10 "各类别召回率:" "${LOG_FILE}" | head -12
fi

echo ""
echo "============================================================"
echo " 全部流程完成!"
echo " 改进方案: 有序回归 + SMOTE-ENN + Optuna + 自适应阈值 + 分层集成"
echo " 日志文件: ${LOG_FILE}"
echo "============================================================"