# 简单机器学习工程项目

一个机器学习工程示例，使用手写数字数据集进行分类任务。相比鸢尾花数据集，这是一个更具挑战性的任务。

## 📁 项目结构

```
simple_test/
├── src/                    # 源代码目录
│   ├── train.py           # 训练脚本
│   └── predict.py         # 预测脚本
├── models/                 # 模型存储目录
│   └── digits_model.joblib # 训练好的手写数字模型
├── dataset/                # 数据集存储目录
├── test/                   # 测试代码目录
│   └── test_train.py      # 单元测试脚本
├── README.md              # 项目说明文件（本文件）
└── baseline.md            # 基线测试结果说明
```

## 🚀 快速开始

### 环境要求

- Python >= 3.7
- scikit-learn
- joblib
- numpy

### 安装依赖

```bash
pip install scikit-learn joblib numpy
```

### 运行训练

```bash
python src/train.py
```

训练完成后，模型会保存到 `models/digits_model.joblib`

### 运行预测

```bash
python src/predict.py
```

### 运行测试

```bash
python test/test_train.py
```

或者使用 verbose 模式查看详细输出：

```bash
python test/test_train.py -v
```

## 📊 数据集说明

本项目使用 **手写数字（Digits）数据集**：

- **样本数量**: 1,797 个
- **特征数量**: 64 个（8x8 灰度图像展平）
- **类别数量**: 10 个（数字 0-9）
- **图像尺寸**: 8x8 像素
- **任务类型**: 多分类任务（10 类）
- **数据特点**: 
  - 每个像素值为 0-16 的整数，表示灰度强度
  - 类别相对平衡
  - 比鸢尾花数据集更具挑战性（更多类别、更高维特征）

## 📝 代码说明

### train.py - 训练脚本

主要功能：
1. 加载手写数字数据集
2. 划分训练集和测试集（80% 训练，20% 测试）
3. 使用随机森林分类器进行训练（200 棵树，最大深度 15）
4. 评估模型性能（准确率、分类报告、混淆矩阵）
5. 保存模型

### predict.py - 预测脚本

主要功能：
1. 加载已保存的模型
2. 对新样本进行预测
3. 输出预测结果和概率分布
4. 以 ASCII 艺术形式显示数字图像

### test_train.py - 测试脚本

包含以下测试类别：
- `TestDataLoading`: 测试数据加载功能
- `TestModelTraining`: 测试模型训练功能
- `TestPrediction`: 测试预测功能
- `TestIntegration`: 集成测试
- `TestDataDistribution`: 测试数据分布

## 🔧 模型配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型类型 | RandomForestClassifier | 随机森林分类器 |
| n_estimators | 200 | 决策树数量 |
| max_depth | 15 | 树的最大深度 |
| test_size | 0.2 | 测试集比例 |
| random_state | 42 | 随机种子 |

## 📈 扩展建议

1. **尝试其他模型**: SVM、神经网络（MLP）、XGBoost 等
2. **特征工程**: PCA 降维、特征选择
3. **超参数调优**: 网格搜索、随机搜索
4. **数据增强**: 旋转、平移图像增加训练数据
5. **深度学习**: 使用 CNN 处理图像分类
6. **可视化**: 添加 t-SNE 降维可视化、学习曲线等

## 📄 许可证

本项目仅用于学习和演示目的。
