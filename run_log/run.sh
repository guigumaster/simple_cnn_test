#!/bin/bash
# ============================================================
# 红酒质量预测 —— XGBoost GPU 加速完整流程 v2.0
# 改进方案：XGBoost GPU加速 + 特征工程 + SMOTE + 加权损失 + 超参数搜索
# ============================================================
set -euo pipefail

# ── 项目根目录（绝对路径）───────────────────────────────────────────
PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/514bde8e-62f3-47f4-b193-f8785ddf8e2b/simple_cnn_test/code/c0ced2c7-4f98-4108-b792-4614af9c7084/simple_cnn_test"

# 子目录
RUN_LOG_DIR="${PROJECT_ROOT}/run_log"
DATASET_DIR="${PROJECT_ROOT}/dataset"
MODEL_DIR="${PROJECT_ROOT}/models"
SRC_DIR="${PROJECT_ROOT}/src"

# 时间戳 & 日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${RUN_LOG_DIR}/run_${TIMESTAMP}.log"

# 确保目录存在
mkdir -p "${RUN_LOG_DIR}" "${MODEL_DIR}"

# ── 日志输出 —— 同时输出到控制台和日志文件 ────────────────────────
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo " 红酒质量预测 —— XGBoost GPU 加速 v2.0"
echo " 项目根目录 : ${PROJECT_ROOT}"
echo " 日志文件   : ${LOG_FILE}"
echo " 时间戳     : ${TIMESTAMP}"
echo "============================================================"
echo ""

# ── 0. 检查硬件环境 ──────────────────────────────────────────────
echo ">>> [0/5] 检查硬件环境..."
nvidia-smi || echo "警告: nvidia-smi 不可用"
echo ""

# ── 1. 数据集准备 ────────────────────────────────────────────────
echo ">>> [1/5] 检查/下载数据集..."
if [ -f "${DATASET_DIR}/winequality.csv" ]; then
    echo "  数据集已存在: ${DATASET_DIR}/winequality.csv"
    echo "  样本数: $(wc -l < "${DATASET_DIR}/winequality.csv") (含表头)"
else
    echo "  下载数据集..."
    python3 "${DATASET_DIR}/download_wine_data.py"
fi
echo ""

# ── 2. 环境检查 ──────────────────────────────────────────────────
echo ">>> [2/5] 检查 Python 依赖..."
echo "  Python: $(python3 --version 2>&1)"
echo "  XGBoost: $(python3 -c 'import xgboost; print(xgboost.__version__)' 2>&1)"
echo "  scikit-learn: $(python3 -c 'import sklearn; print(sklearn.__version__)' 2>&1)"
echo "  pandas: $(python3 -c 'import pandas; print(pandas.__version__)' 2>&1)"
echo "  numpy: $(python3 -c 'import numpy; print(numpy.__version__)' 2>&1)"
echo "  joblib: $(python3 -c 'import joblib; print(joblib.__version__)' 2>&1)"
echo "  imbalanced-learn: $(python3 -c 'import imblearn; print(imblearn.__version__)' 2>&1)"
echo ""

# ── 3. 模型训练（GPU 加速） ────────────────────────────────────────
echo ">>> [3/5] 开始训练 XGBoost 模型 (GPU 加速)..."
echo "  训练脚本: ${SRC_DIR}/train.py"
echo "  开始时间: $(date)"
echo ""

# 调用绝对路径下的 Python 脚本
python3 "${SRC_DIR}/train.py"

echo ""
echo "  结束时间: $(date)"
echo "  训练完成!"
echo ""

# ── 4. 模型预测 ──────────────────────────────────────────────────
echo ">>> [4/5] 加载模型进行预测..."
echo "  预测脚本: ${SRC_DIR}/predict.py"
echo "  开始时间: $(date)"
echo ""

python3 "${SRC_DIR}/predict.py"

echo ""
echo "  结束时间: $(date)"
echo ""

# ── 5. 输出结果摘要 ──────────────────────────────────────────────
echo ">>> [5/5] 输出结果摘要..."
MODEL_FILE="${MODEL_DIR}/wine_model.joblib"
if [ -f "${MODEL_FILE}" ]; then
    MODEL_SIZE=$(du -h "${MODEL_FILE}" | cut -f1)
    echo "  模型文件: ${MODEL_FILE} (${MODEL_SIZE})"
fi
echo "  日志文件: ${LOG_FILE}"
echo ""

# 从日志中提取准确率
if [ -f "${LOG_FILE}" ]; then
    # 查找准确率行（来自 predict.py 的输出）
    ACC_LINE=$(grep -E "测试集准确率" "${LOG_FILE}" | tail -1)
    echo "  最新准确率: ${ACC_LINE:-N/A}"
fi

echo ""
echo "============================================================"
echo " 全部流程完成!"
echo "============================================================"