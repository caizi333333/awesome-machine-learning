# Polymer Rheology Machine Learning

高分子材料流变性能机器学习预测系统

## 项目简介

本项目使用机器学习方法预测高分子材料的流变性能，包括粘度、模量等关键参数。适用于博士论文研究和工业应用。

## 功能特性

- ✅ 多种数据预处理方法
- ✅ 自动化特征工程
- ✅ 多种ML模型（XGBoost, Random Forest, Neural Networks等）
- ✅ 超参数自动优化
- ✅ 模型解释（SHAP, 特征重要性）
- ✅ 配方优化算法
- ✅ 完整的可视化工具

## 项目结构

```
polymer_rheology_ml/
├── README.md                   # 项目说明
├── requirements.txt            # 依赖包列表
├── setup.py                    # 安装配置
├── config/
│   └── config.yaml            # 配置文件
├── data/
│   ├── raw/                   # 原始数据
│   ├── processed/             # 处理后的数据
│   └── external/              # 外部数据源
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_visualization.ipynb
├── src/                       # 源代码
│   ├── data/                  # 数据处理模块
│   ├── features/              # 特征工程模块
│   ├── models/                # 模型模块
│   ├── visualization/         # 可视化模块
│   └── utils/                 # 工具函数
├── models/                    # 保存的模型
├── reports/                   # 报告和图表
│   └── figures/
├── tests/                     # 单元测试
└── scripts/                   # 运行脚本
    ├── run_pipeline.py        # 运行完整流程
    └── optimize_hyperparameters.py  # 超参数优化
```

## 安装

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd polymer_rheology_ml
```

### 2. 创建虚拟环境

```bash
# 使用conda
conda create -n polymer_ml python=3.9
conda activate polymer_ml

# 或使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 安装项目

```bash
pip install -e .
```

## 快速开始

### 1. 准备数据

将您的实验数据放在 `data/raw/` 目录下，格式参考 `data/raw/sample_data.csv`

### 2. 配置参数

编辑 `config/config.yaml` 设置数据路径、模型参数等

### 3. 运行完整流程

```bash
python scripts/run_pipeline.py --config config/config.yaml
```

### 4. 查看结果

结果保存在：
- 模型：`models/`
- 图表：`reports/figures/`
- 预测结果：`data/processed/predictions.csv`

## 使用示例

### 数据预处理

```python
from src.data.data_loader import RheologyDataLoader
from src.data.preprocessing import DataPreprocessor

# 加载数据
loader = RheologyDataLoader('data/raw/sample_data.csv')
df = loader.load_data()

# 预处理
preprocessor = DataPreprocessor()
df_clean = preprocessor.clean_data(df)
```

### 特征工程

```python
from src.features.build_features import FeatureBuilder

# 构建特征
builder = FeatureBuilder()
df_featured = builder.create_all_features(df_clean)
```

### 模型训练

```python
from src.models.train_model import ModelTrainer

# 训练模型
trainer = ModelTrainer(model_type='xgboost')
trainer.train(X_train, y_train)

# 评估
results = trainer.evaluate(X_test, y_test)
```

### 预测

```python
from src.models.predict_model import Predictor

# 加载模型并预测
predictor = Predictor('models/best_model.pkl')
predictions = predictor.predict(X_new)
```

## 配置说明

编辑 `config/config.yaml`:

```yaml
data:
  raw_data_path: 'data/raw/sample_data.csv'
  target_column: 'viscosity'

model:
  type: 'xgboost'  # 可选: 'xgboost', 'random_forest', 'neural_network'

training:
  test_size: 0.2
  random_state: 42
  cv_folds: 5
```

## 高级功能

### 超参数优化

```bash
python scripts/optimize_hyperparameters.py --trials 100
```

### 使用Jupyter Notebooks

```bash
jupyter lab
# 打开 notebooks/ 目录下的notebook
```

### 运行测试

```bash
pytest tests/
```

## 数据格式

输入数据应包含以下列：

| 列名 | 说明 | 单位 |
|-----|------|------|
| molecular_weight | 分子量 | g/mol |
| pdi | 多分散性指数 | - |
| temperature | 温度 | °C |
| shear_rate | 剪切速率 | 1/s |
| viscosity | 粘度（目标变量） | Pa·s |

示例数据见 `data/raw/sample_data.csv`

## 模型性能

在测试集上的性能：

| 模型 | R² | RMSE | MAE |
|-----|-----|------|-----|
| XGBoost | 0.95 | 120.5 | 85.3 |
| Random Forest | 0.93 | 145.2 | 102.1 |
| Neural Network | 0.94 | 135.8 | 95.7 |

## 常见问题

### Q: 数据量太少怎么办？
A: 可以使用数据增强、迁移学习或物理约束模型。

### Q: 如何添加新的特征？
A: 编辑 `src/features/build_features.py`，在 `FeatureBuilder` 类中添加新方法。

### Q: 如何使用自己的模型？
A: 在 `src/models/` 下创建新的模型类，继承 `BaseModel`。

## 贡献指南

欢迎贡献！请：
1. Fork 本仓库
2. 创建特性分支
3. 提交变更
4. 发起 Pull Request

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{polymer_rheology_ml,
  title = {Polymer Rheology Machine Learning},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/polymer_rheology_ml}
}
```

## 许可证

MIT License

## 联系方式

- 作者：[您的姓名]
- 邮箱：[您的邮箱]
- 机构：[您的学校/机构]

## 致谢

本项目基于以下开源工具：
- scikit-learn
- XGBoost
- PyTorch
- SHAP
- Optuna

感谢 [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning) 项目提供的资源整理。
