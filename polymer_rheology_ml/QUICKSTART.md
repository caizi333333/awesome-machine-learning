# 快速开始指南

本指南将帮助您在15分钟内运行第一个流变性能预测模型。

## 1. 环境准备（5分钟）

###  克隆项目

```bash
cd /your/workspace
# 项目已经存在于 polymer_rheology_ml/
```

### 创建虚拟环境

```bash
# 进入项目目录
cd polymer_rheology_ml

# 创建conda环境
conda create -n polymer_ml python=3.9 -y
conda activate polymer_ml

# 或使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 生成示例数据（1分钟）

```bash
# 生成1000个样本数据点
python -c "
import sys
sys.path.append('.')
from src.data.data_loader import create_sample_data
create_sample_data(1000, 'data/raw/sample_data.csv')
print('Sample data created!')
"
```

或者直接运行：

```bash
python src/data/data_loader.py
```

## 3. 运行完整流程（5分钟）

```bash
python scripts/run_pipeline.py
```

这将执行以下步骤：
1. ✅ 加载数据
2. ✅ 数据清洗
3. ✅ 特征工程
4. ✅ 特征选择
5. ✅ 数据标准化
6. ✅ 模型训练
7. ✅ 模型评估
8. ✅ 生成可视化图表
9. ✅ 保存模型

## 4. 查看结果

### 模型性能

查看控制台输出，您会看到：

```
Test Metrics:
{
  'mse': 12345.67,
  'rmse': 111.11,
  'mae': 85.33,
  'r2': 0.95,
  'mape': 5.23
}
```

### 可视化图表

生成的图表保存在 `reports/figures/`:
- `predictions.png` - 预测值 vs 真实值
- `feature_importance.png` - 特征重要性

### 保存的模型

训练好的模型保存在：
- `models/best_model.pkl`

## 5. 使用模型进行预测

```python
from src.models.predict_model import Predictor
import pandas as pd

# 加载模型
predictor = Predictor(model_path='models/best_model.pkl')

# 准备新数据（确保特征与训练时一致）
new_data = pd.DataFrame({
    'molecular_weight': [150000],
    'pdi': [2.0],
    'temperature': [200],
    'shear_rate': [10],
    'concentration': [25]
})

# 预测
viscosity = predictor.predict(new_data)
print(f"预测粘度: {viscosity[0]:.2f} Pa·s")
```

## 6. 使用Jupyter Notebook（推荐）

```bash
# 启动Jupyter Lab
jupyter lab
```

打开 `notebooks/` 目录下的notebook：
- `01_data_exploration.ipynb` - 数据探索
- `02_feature_engineering.ipynb` - 特征工程
- `03_model_training.ipynb` - 模型训练
- `04_results_visualization.ipynb` - 结果可视化

## 常见问题

### Q1: 依赖安装失败？

```bash
# 尝试清华镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: 内存不足？

减少样本数量：
```python
create_sample_data(500, 'data/raw/sample_data.csv')  # 减少到500
```

### Q3: 想使用自己的数据？

将您的CSV文件放在 `data/raw/` 目录，然后修改 `config/config.yaml`:

```yaml
data:
  raw_data_path: 'data/raw/my_data.csv'
  target_column: 'viscosity'  # 您的目标变量列名
```

### Q4: 如何调整模型参数？

编辑 `config/config.yaml` 中的模型参数：

```yaml
model:
  type: 'xgboost'
  xgboost:
    n_estimators: 200  # 增加树的数量
    learning_rate: 0.05  # 降低学习率
    max_depth: 8  # 增加树的深度
```

## 下一步

1. **探索数据**: 运行 `notebooks/01_data_exploration.ipynb`
2. **优化模型**: 运行超参数优化
3. **添加特征**: 在 `src/features/build_features.py` 中添加领域特征
4. **尝试其他模型**: 修改 `config.yaml` 中的 `model.type`

## 项目结构

```
polymer_rheology_ml/
├── data/              # 数据目录
│   ├── raw/          # 原始数据
│   └── processed/    # 处理后的数据
├── src/              # 源代码
│   ├── data/         # 数据处理
│   ├── features/     # 特征工程
│   ├── models/       # 模型
│   └── visualization/ # 可视化
├── models/           # 保存的模型
├── reports/          # 报告和图表
├── notebooks/        # Jupyter notebooks
├── scripts/          # 运行脚本
└── config/           # 配置文件
```

## 支持

遇到问题？
- 查看详细文档：`README.md`
- 查看代码示例：`polymer_rheology_code_examples.md`
- 查看研究方案：`polymer_rheology_ml_research_plan.md`

祝您使用愉快！ 🎉
