#!/bin/bash
#==============================================================================
# run.sh
# 基于 GPU 加速 XGBoost 的多策略融合方案 - 完整执行流程
#
# 改进方案：
#   1. GPU 加速 XGBoost 替代 RandomForest
#   2. 补全缺失特征 'wine_type'（红酒/白酒判别信息）
#   3. 自定义类别权重 + SMOTE 过采样处理类别不平衡
#   4. 引入交互特征工程（PolynomialFeatures）
#
# 预期效果：
#   - 准确率从 62.69% 提升至 68–72%
#   - 稀有类别（质量 3、4、9）的召回率和宏平均 F1 显著改善
#
# 使用方法：
#   bash run_log/run.sh              # 完整流程
#   bash run_log/run.sh --skip-download  # 跳过数据下载
#   bash run_log/run.sh --train-only     # 仅训练
#   bash run_log/run.sh --predict-only   # 仅预测
#==============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 项目根目录（绝对路径）
# ---------------------------------------------------------------------------
PROJECT_ROOT="/inspire/cpfs/project/sais-ai-for-science-code/public/mession/running_location/5cc9487a-21ac-48c0-b416-7732c36b6008/simple_cnn_test/code/e5ee4e74-06ad-406c-9c39-8f539ed8efda/simple_cnn_test"

DATASET_DIR="${PROJECT_ROOT}/dataset"
SRC_DIR="${PROJECT_ROOT}/src"
MODEL_DIR="${PROJECT_ROOT}/models"
LOG_DIR="${PROJECT_ROOT}/run_log"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

# ---------------------------------------------------------------------------
# 命令行参数解析
# ---------------------------------------------------------------------------
SKIP_DOWNLOAD=false
TRAIN_ONLY=false
PREDICT_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --train-only)    TRAIN_ONLY=true ;;
        --predict-only)  PREDICT_ONLY=true ;;
        --help)
            echo "用法: bash $0 [选项]"
            echo "选项:"
            echo "  --skip-download   跳过数据下载步骤"
            echo "  --train-only      仅执行训练"
            echo "  --predict-only    仅执行预测"
            echo "  --help            显示帮助信息"
            exit 0
            ;;
    esac
done

# ---------------------------------------------------------------------------
# 日志函数
# ---------------------------------------------------------------------------
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

log_separator() {
    echo "" | tee -a "$LOG_FILE"
    echo "================================================================================" | tee -a "$LOG_FILE"
    echo "$*" | tee -a "$LOG_FILE"
    echo "================================================================================" | tee -a "$LOG_FILE"
}

# ---------------------------------------------------------------------------
# 环境检查
# ---------------------------------------------------------------------------
check_environment() {
    log_separator "[环境检查]"

    # 项目根目录
    if [ ! -d "$PROJECT_ROOT" ]; then
        echo "错误：项目根目录不存在：$PROJECT_ROOT"
        exit 1
    fi
    log "项目根目录: ${PROJECT_ROOT}"

    # Python
    PYTHON_CMD=$(command -v python3 || command -v python)
    if [ -z "$PYTHON_CMD" ]; then
        echo "错误：未找到 python3/python 命令"
        exit 1
    fi
    log "Python: $($PYTHON_CMD --version 2>&1)"

    # 检查 GPU
    if command -v nvidia-smi &>/dev/null; then
        log "GPU 状态:"
        nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>&1 | while IFS= read -r line; do
            log "  $line"
        done
    else
        log "警告：未检测到 nvidia-smi，GPU 可能不可用"
    fi

    # 检查关键 Python 包
    log "Python 包检查:"
    for pkg in xgboost scikit-learn pandas numpy joblib; do
        if $PYTHON_CMD -c "import $pkg" 2>/dev/null; then
            ver=$($PYTHON_CMD -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
            log "  $pkg == $ver"
        else
            log "  $pkg - 未安装"
        fi
    done

    # 检查 imbalanced-learn（用于 SMOTE）
    if $PYTHON_CMD -c "import imblearn" 2>/dev/null; then
        log "  imbalanced-learn (imblearn) == $($PYTHON_CMD -c 'import imblearn; print(imblearn.__version__)' 2>/dev/null || echo 'ok')"
    else
        log "  imbalanced-learn (imblearn) - 未安装，需要安装"
    fi

    log "环境检查完成"
}

# ---------------------------------------------------------------------------
# 安装依赖
# ---------------------------------------------------------------------------
install_dependencies() {
    log_separator "[安装依赖]"

    # 安装 imbalanced-learn（用于 SMOTE 过采样）
    if ! $PYTHON_CMD -c "import imblearn" 2>/dev/null; then
        log "安装 imbalanced-learn..."
        pip install imbalanced-learn -q 2>&1 | tee -a "$LOG_FILE"
        log "imbalanced-learn 安装完成"
    else
        log "imbalanced-learn 已安装，跳过"
    fi

    # 确认 xgboost GPU 支持
    log "验证 XGBoost GPU 支持..."
    $PYTHON_CMD -c "
import xgboost as xgb
import numpy as np
dtrain = xgb.DMatrix(np.random.rand(10, 5), label=np.random.randint(0, 3, 10))
params = {'tree_method': 'hist', 'device': 'cuda', 'objective': 'multi:softprob', 'num_class': 3}
model = xgb.train(params, dtrain, num_boost_round=2)
print('XGBoost GPU 训练验证通过!')
" 2>&1 | tee -a "$LOG_FILE"

    log "依赖安装完成"
}

# ---------------------------------------------------------------------------
# Step 1: 下载数据
# ---------------------------------------------------------------------------
download_data() {
    log_separator "[Step 1/4] 下载数据集"

    local data_file="${DATASET_DIR}/winequality.csv"
    if [ -f "$data_file" ]; then
        log "数据集已存在：${data_file}"
        log "如需重新下载，请先删除该文件"
        return
    fi

    log "下载红酒和白酒质量数据集（UCI Wine Quality）..."
    $PYTHON_CMD "${DATASET_DIR}/download_wine_data.py" 2>&1 | tee -a "$LOG_FILE"

    if [ -f "$data_file" ]; then
        log "数据下载成功：${data_file}"
    else
        echo "错误：数据下载失败"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Step 2: 训练模型
# ---------------------------------------------------------------------------
train_model() {
    log_separator "[Step 2/4] 训练模型（GPU 加速 XGBoost）"

    log "训练参数:"
    log "  模型: XGBoost (GPU: cuda)"
    log "  特征: 11 理化特征 + wine_type + 二阶交互特征"
    log "  过采样: SMOTE"
    log "  类别权重: balanced"
    log "  目标: 提升准确率至 68-72% + 改善稀有类别召回率"

    $PYTHON_CMD "${SRC_DIR}/train.py" 2>&1 | tee -a "$LOG_FILE"

    # 检查模型是否生成
    if [ -f "${MODEL_DIR}/wine_model.joblib" ]; then
        log "模型训练成功！"
        log "模型文件：${MODEL_DIR}/wine_model.joblib"
    else
        echo "错误：模型训练失败，未生成 wine_model.joblib"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Step 3: 预测评估
# ---------------------------------------------------------------------------
predict_model() {
    log_separator "[Step 3/4] 模型预测与评估"

    if [ ! -f "${MODEL_DIR}/wine_model.joblib" ]; then
        echo "错误：模型文件不存在，请先执行训练步骤"
        exit 1
    fi

    log "加载模型并评估测试集..."
    $PYTHON_CMD "${SRC_DIR}/predict.py" 2>&1 | tee -a "$LOG_FILE"

    log "预测评估完成"
}

# ---------------------------------------------------------------------------
# Step 4: 结果摘要
# ---------------------------------------------------------------------------
print_summary() {
    log_separator "[Step 4/4] 结果汇总"

    log "改进方案总结:"
    log "  ✅ GPU 加速 XGBoost (device=cuda)"
    log "  ✅ 补全 wine_type 特征（红酒/白酒判别）"
    log "  ✅ SMOTE 过采样处理类别不平衡"
    log "  ✅ 自定义类别权重（balanced 策略）"
    log "  ✅ 二阶交互特征工程（PolynomialFeatures）"
    log ""
    log "日志文件: ${LOG_FILE}"
    log "模型文件: ${MODEL_DIR}/wine_model.joblib"
    log "训练数据: ${MODEL_DIR}/train_data.joblib"
    log "测试数据: ${MODEL_DIR}/test_data.joblib"

    # 如果有测试集评估结果，提取准确率
    if [ -f "${MODEL_DIR}/test_data.joblib" ]; then
        log ""
        log "从日志中提取最新评估指标..."
        grep -E "(准确率.*Accuracy|宏平均 F1|加权平均 F1)" "${LOG_FILE}" 2>/dev/null | head -5 | while IFS= read -r line; do
            log "  $line"
        done
    fi

    log ""
    log "执行完成！"
}

# ============================================================================
# 主流程
# ============================================================================

# 初始化日志
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

log "============================================"
log "基于 GPU 加速 XGBoost 的多策略融合方案"
log "项目: 红酒质量预测 (UCI Wine Quality)"
log "根目录: ${PROJECT_ROOT}"
log "时间戳: ${TIMESTAMP}"
log "============================================"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 执行各步骤
check_environment
install_dependencies

if [ "$PREDICT_ONLY" = false ]; then
    if [ "$SKIP_DOWNLOAD" = false ]; then
        download_data
    else
        log "跳过数据下载 (--skip-download)"
    fi
    train_model
fi

if [ "$TRAIN_ONLY" = false ]; then
    predict_model
fi

print_summary

log_separator "[全部流程结束]"