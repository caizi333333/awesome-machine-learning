# 如何使用这个项目模板

## 🎯 项目已创建完成！

您现在拥有一个完整的、生产就绪的Python机器学习项目，专门为高分子流变性能研究设计。

---

## 📦 项目内容

### 已创建的文件（31个）

```
polymer_rheology_ml/
├── 📄 README.md                      ← 主文档（从这里开始）
├── 📄 QUICKSTART.md                  ← 15分钟快速开始
├── 📄 PROJECT_SUMMARY.md             ← 项目总结
├── 📄 HOW_TO_USE.md                  ← 本文件
├── 📄 requirements.txt               ← 依赖包列表
├── 📄 setup.py                       ← 安装配置
├── 📄 .gitignore                    ← Git配置
├── 🐍 generate_project.py            ← 项目生成脚本
│
├── 📁 config/
│   └── config.yaml                  ← 核心配置文件
│
├── 📁 src/                          ← 源代码（可直接使用）
│   ├── data/                        ← 数据处理
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   ├── features/                    ← 特征工程
│   │   └── build_features.py
│   ├── models/                      ← 模型训练/预测/评估
│   │   ├── train_model.py
│   │   ├── predict_model.py
│   │   └── evaluate_model.py
│   ├── visualization/               ← 可视化
│   │   └── visualize.py
│   └── utils/                       ← 工具函数
│       └── helpers.py
│
├── 📁 scripts/
│   └── run_pipeline.py              ← 主运行脚本
│
├── 📁 notebooks/                    ← Jupyter notebooks
│   └── README.md
│
├── 📁 tests/                        ← 单元测试
│   └── test_data_loader.py
│
└── 📁 data/, models/, reports/      ← 数据和输出目录
```

---

## 🚀 立即开始（3种方式）

### 方式1: 最快速（推荐初学者）

```bash
# 1. 进入项目目录
cd awesome-machine-learning/polymer_rheology_ml

# 2. 创建虚拟环境
conda create -n polymer_ml python=3.9
conda activate polymer_ml

# 3. 安装依赖
pip install -r requirements.txt

# 4. 生成示例数据并运行
python src/data/data_loader.py  # 生成示例数据
python scripts/run_pipeline.py   # 运行完整流程

# 5. 查看结果
ls reports/figures/              # 查看生成的图表
ls models/                       # 查看保存的模型
```

**预期结果**：
- ✅ 训练完成，显示R² > 0.90
- ✅ 生成预测图和特征重要性图
- ✅ 模型保存到 `models/best_model.pkl`

---

### 方式2: 使用Jupyter Notebook（推荐深度探索）

```bash
# 1. 安装依赖（同上）

# 2. 启动Jupyter Lab
jupyter lab

# 3. 在浏览器中打开
# 访问 notebooks/ 目录
# 按顺序运行：
#   - 01_data_exploration.ipynb
#   - 02_feature_engineering.ipynb
#   - 03_model_training.ipynb
#   - 04_results_visualization.ipynb
```

---

### 方式3: 使用自己的数据

```bash
# 1. 准备数据
# 将您的CSV文件放到 data/raw/my_data.csv
# 格式要求：包含 molecular_weight, pdi, temperature, shear_rate, viscosity等列

# 2. 修改配置
# 编辑 config/config.yaml:
data:
  raw_data_path: 'data/raw/my_data.csv'
  target_column: 'viscosity'  # 您的目标变量

# 3. 运行
python scripts/run_pipeline.py
```

---

## 📚 核心文档导航

### 新手必读
1. **README.md** - 项目总览和安装说明
2. **QUICKSTART.md** - 快速开始指南
3. **PROJECT_SUMMARY.md** - 详细功能说明

### 深入学习
4. **polymer_rheology_ml_research_plan.md** (在上级目录) - 研究方案
5. **polymer_rheology_code_examples.md** (在上级目录) - 代码示例

### 配置参考
6. **config/config.yaml** - 所有参数说明
7. **requirements.txt** - 依赖包列表

---

## 🔧 核心功能使用

### 1. 数据加载

```python
from src.data.data_loader import RheologyDataLoader

loader = RheologyDataLoader('data/raw/sample_data.csv')
df = loader.load_data()
print(df.head())
```

### 2. 数据预处理

```python
from src.data.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
df_clean = preprocessor.clean_data(df)
df_normalized = preprocessor.normalize_data(df_clean)
```

### 3. 特征工程

```python
from src.features.build_features import FeatureBuilder

builder = FeatureBuilder()
df_featured = builder.create_all_features(df)
```

### 4. 模型训练

```python
from src.models.train_model import ModelTrainer

trainer = ModelTrainer(model_type='xgboost')
trainer.train(X_train, y_train)
trainer.save_model('models/my_model.pkl')
```

### 5. 模型预测

```python
from src.models.predict_model import Predictor

predictor = Predictor(model_path='models/my_model.pkl')
predictions = predictor.predict(X_new)
```

### 6. 模型评估

```python
from src.models.evaluate_model import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate(y_true, y_pred)
print(metrics)  # {'rmse': ..., 'r2': ..., 'mae': ...}
```

### 7. 可视化

```python
from src.visualization.visualize import Visualizer

viz = Visualizer()
viz.plot_predictions(y_test, y_pred, save_path='reports/figures/pred.png')
viz.plot_feature_importance(model, features)
```

---

## ⚙️ 自定义配置

### 修改模型参数

编辑 `config/config.yaml`:

```yaml
model:
  type: 'xgboost'  # 或 'random_forest', 'lightgbm', 'svr'

  xgboost:
    n_estimators: 200      # 树的数量
    learning_rate: 0.05    # 学习率
    max_depth: 8           # 树的深度
    min_child_weight: 3
    subsample: 0.9
    colsample_bytree: 0.9
```

### 修改特征工程

编辑 `config/config.yaml`:

```yaml
feature_engineering:
  create_domain_features: true
  create_polynomial: true
  polynomial_degree: 2
  create_interactions: true
  interaction_pairs:
    - ['temperature', 'shear_rate']
    - ['molecular_weight', 'pdi']
```

### 修改数据预处理

编辑 `config/config.yaml`:

```yaml
preprocessing:
  missing_value_strategy: 'mean'  # 'mean', 'median', 'drop'
  outlier_method: 'iqr'           # 'iqr', 'zscore', 'none'
  outlier_threshold: 1.5
  normalization_method: 'standard' # 'standard', 'minmax', 'robust'
```

---

## 📊 输出说明

### 运行后生成的文件

```
polymer_rheology_ml/
├── data/processed/
│   └── predictions.csv           # 预测结果
│
├── models/
│   └── best_model.pkl            # 保存的模型（可重复使用）
│
└── reports/figures/
    ├── predictions.png           # 预测vs真实值图
    └── feature_importance.png    # 特征重要性图
```

### 控制台输出示例

```
Step 1: Loading data
Data loaded successfully. Shape: (1000, 6)

Step 2: Preprocessing data
Removed 15 outliers
Missing values handled

Step 3: Feature engineering
Created 25 new features

Step 4: Feature selection
Selected 18 features

Step 5: Splitting data
Split completed: train=700, val=100, test=200

Step 6: Training model
XGBoost model training...
Training completed

Step 7: Evaluating model

Test Metrics:
{'mse': 12456.78, 'rmse': 111.61, 'mae': 85.34, 'r2': 0.9523, 'mape': 5.23}

Pipeline completed successfully!
```

---

## 🎓 博士论文应用场景

### 场景1: 数据驱动的性能预测

```python
# 使用历史数据训练模型
trainer = ModelTrainer(model_type='xgboost')
trainer.train(X_historical, y_viscosity)

# 预测新配方的性能
new_formulation = pd.DataFrame({...})
predicted_viscosity = predictor.predict(new_formulation)
```

### 场景2: 配方优化

```python
# 在 src/utils/optimization.py 中（需要您添加）
from scipy.optimize import differential_evolution

def optimize_formulation(model, target_viscosity, bounds):
    def objective(x):
        pred = model.predict(x.reshape(1, -1))
        return (pred - target_viscosity) ** 2

    result = differential_evolution(objective, bounds)
    return result.x  # 最优参数
```

### 场景3: 参数敏感性分析

```python
# 使用SHAP分析
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=features)
```

---

## 🐛 故障排除

### 问题1: ModuleNotFoundError

```bash
# 确保在正确的环境中
conda activate polymer_ml

# 重新安装依赖
pip install -r requirements.txt
```

### 问题2: 路径错误

```python
# 在notebooks或scripts中
import sys
sys.path.append('..')  # 添加项目根目录
```

### 问题3: YAML配置错误

```bash
# 检查缩进（必须使用空格，不要用Tab）
# 检查路径是否正确
```

### 问题4: 内存不足

```yaml
# 修改config.yaml，减少数据量
data:
  test_size: 0.3  # 增加测试集，减少训练集

model:
  xgboost:
    n_estimators: 50  # 减少树的数量
```

---

## 📈 下一步

### 学习路径

1. **Week 1-2**: 熟悉项目结构，运行示例
2. **Week 3-4**: 使用自己的数据，调整参数
3. **Week 5-6**: 添加自定义特征和模型
4. **Week 7-8**: 进行超参数优化
5. **Week 9+**: 撰写论文，生成图表

### 扩展功能

- [ ] 添加深度学习模型（PyTorch）
- [ ] 添加SHAP集成
- [ ] 添加超参数优化脚本
- [ ] 添加交叉验证可视化
- [ ] 添加模型对比报告
- [ ] 创建Web界面（Streamlit）

---

## 💡 最佳实践

### 1. 版本控制
```bash
git add .
git commit -m "描述您的更改"
```

### 2. 实验记录
创建 `experiments/` 目录，保存每次实验的配置和结果

### 3. 代码注释
在修改代码时添加中文注释，方便论文写作时回顾

### 4. 定期备份
定期备份 `data/`, `models/`, `reports/`

---

## 📞 获取帮助

1. **查看文档**: README.md, QUICKSTART.md
2. **查看代码**: 所有代码都有详细注释
3. **查看示例**: polymer_rheology_code_examples.md
4. **查看研究方案**: polymer_rheology_ml_research_plan.md

---

## ✅ 检查清单

在开始使用前，确保：

- [ ] 已安装Python 3.8+
- [ ] 已创建虚拟环境
- [ ] 已安装所有依赖（`pip install -r requirements.txt`）
- [ ] 已阅读README.md
- [ ] 已成功运行示例（`python scripts/run_pipeline.py`）
- [ ] 了解项目结构
- [ ] 知道如何修改配置文件

---

## 🎉 恭喜！

您现在拥有一个专业的、可扩展的机器学习项目模板。

**祝您的博士研究顺利！** 🎓

如有任何问题，请参考文档或查看代码中的注释。

---

**最后更新**: 2025-10-24
