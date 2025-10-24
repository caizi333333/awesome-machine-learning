# Jupyter Notebooks

本目录包含用于交互式数据分析和模型开发的Jupyter notebooks。

## Notebooks列表

### 01_data_exploration.ipynb
**数据探索与分析**

- 加载和查看数据
- 统计描述
- 数据质量检查
- 可视化分布
- 相关性分析

**建议运行时间**: 10-15分钟

---

### 02_feature_engineering.ipynb
**特征工程**

- 创建领域特征
- 多项式特征
- 交互特征
- 特征选择
- 特征重要性分析

**建议运行时间**: 15-20分钟

---

### 03_model_training.ipynb
**模型训练与对比**

- 多种模型训练
- 超参数调优
- 交叉验证
- 模型对比
- 选择最佳模型

**建议运行时间**: 20-30分钟

---

### 04_results_visualization.ipynb
**结果可视化**

- 预测结果可视化
- 特征重要性可视化
- SHAP分析
- 残差分析
- 生成论文图表

**建议运行时间**: 10-15分钟

---

## 使用说明

### 启动Jupyter Lab

```bash
# 在项目根目录运行
jupyter lab
```

### 运行顺序

建议按照编号顺序运行notebooks：
1. `01_data_exploration.ipynb` - 了解数据
2. `02_feature_engineering.ipynb` - 构建特征
3. `03_model_training.ipynb` - 训练模型
4. `04_results_visualization.ipynb` - 分析结果

### 创建新的Notebook

```python
# Notebook模板代码结构

# 1. 导入库
import sys
sys.path.append('..')  # 添加项目路径

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.data_loader import RheologyDataLoader
from src.data.preprocessing import DataPreprocessor
# ... 其他导入

# 2. 配置
%matplotlib inline
plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', None)

# 3. 加载数据
loader = RheologyDataLoader('../data/raw/sample_data.csv')
df = loader.load_data()

# 4. 分析代码
# ...

# 5. 保存结果
# df.to_csv('../data/processed/processed_data.csv', index=False)
```

## 注意事项

1. **路径问题**: notebooks运行在`notebooks/`目录，访问项目文件需要使用`../`
2. **内核选择**: 确保使用`polymer_ml`虚拟环境的内核
3. **保存结果**: 中间结果保存到`data/processed/`
4. **图表保存**: 生成的图表保存到`reports/figures/`

## 提示

- 使用 `Shift + Enter` 运行单元格
- 使用 `Esc + A` 在上方插入单元格
- 使用 `Esc + B` 在下方插入单元格
- 使用 `Esc + DD` 删除单元格
- 使用 `Ctrl + /` 注释/取消注释代码

## 导出为Python脚本

```bash
# 导出notebook为.py文件
jupyter nbconvert --to script notebook_name.ipynb
```

## 论文中使用图表

生成高质量图表的建议：

```python
# 设置图表样式
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Arial'

# 保存图表
plt.savefig('../reports/figures/figure_name.png',
            dpi=300, bbox_inches='tight', transparent=False)
plt.savefig('../reports/figures/figure_name.pdf',  # PDF for LaTeX
            bbox_inches='tight')
```
