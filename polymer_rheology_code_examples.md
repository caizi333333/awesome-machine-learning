# 高分子流变性能机器学习 - Python代码示例

本文档提供完整的代码示例，帮助您快速上手机器学习在高分子流变性能研究中的应用。

---

## 目录
1. [环境配置](#1-环境配置)
2. [数据预处理](#2-数据预处理)
3. [特征工程](#3-特征工程)
4. [模型训练](#4-模型训练)
5. [模型评估](#5-模型评估)
6. [模型解释](#6-模型解释)
7. [优化算法](#7-优化算法)
8. [完整案例](#8-完整案例)

---

## 1. 环境配置

### 1.1 创建虚拟环境

```bash
# 使用conda创建环境
conda create -n polymer_ml python=3.9
conda activate polymer_ml

# 或使用venv
python -m venv polymer_ml_env
source polymer_ml_env/bin/activate  # Linux/Mac
# polymer_ml_env\Scripts\activate  # Windows
```

### 1.2 安装依赖包

```bash
# requirements.txt
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scipy==1.11.1
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install scikit-learn==1.3.0
pip install xgboost==1.7.6
pip install lightgbm==4.0.0
pip install torch==2.0.1
pip install shap==0.42.1
pip install optuna==3.3.0
pip install joblib==1.3.2
pip install jupyterlab==4.0.3
```

```python
# 验证安装
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import torch

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"XGBoost: {xgb.__version__}")
print(f"PyTorch: {torch.__version__}")
```

---

## 2. 数据预处理

### 2.1 数据加载

```python
import pandas as pd
import numpy as np
from pathlib import Path

class RheologyDataLoader:
    """流变数据加载器"""

    def __init__(self, data_path):
        self.data_path = Path(data_path)

    def load_viscosity_data(self, filename):
        """加载粘度数据"""
        df = pd.read_csv(self.data_path / filename)
        print(f"数据维度: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        return df

    def load_modulus_data(self, filename):
        """加载模量数据"""
        df = pd.read_csv(self.data_path / filename)
        return df

# 使用示例
loader = RheologyDataLoader("data/raw")
df = loader.load_viscosity_data("viscosity_data.csv")

# 查看数据
print(df.head())
print(df.describe())
print(df.info())
```

### 2.2 数据清洗

```python
class DataCleaner:
    """数据清洗器"""

    @staticmethod
    def remove_outliers(df, column, method='iqr', threshold=1.5):
        """去除异常值"""
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)

        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[column]))
            mask = z_scores < threshold

        print(f"移除了 {(~mask).sum()} 个异常值")
        return df[mask]

    @staticmethod
    def handle_missing_values(df, strategy='mean'):
        """处理缺失值"""
        from sklearn.impute import SimpleImputer

        if strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        return df

    @staticmethod
    def check_data_quality(df):
        """检查数据质量"""
        print("=" * 50)
        print("数据质量报告")
        print("=" * 50)
        print(f"总样本数: {len(df)}")
        print(f"\n缺失值统计:")
        print(df.isnull().sum())
        print(f"\n重复样本数: {df.duplicated().sum()}")
        return df

# 使用示例
cleaner = DataCleaner()
df_clean = cleaner.handle_missing_values(df)
df_clean = cleaner.remove_outliers(df_clean, 'viscosity', method='iqr')
cleaner.check_data_quality(df_clean)
```

### 2.3 数据标准化

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class DataNormalizer:
    """数据标准化器"""

    def __init__(self, method='standard'):
        if method == 'standard':
            self.scaler = StandardScaler()  # Z-score标准化
        elif method == 'minmax':
            self.scaler = MinMaxScaler()    # 0-1归一化
        elif method == 'robust':
            self.scaler = RobustScaler()    # 鲁棒标准化

    def fit_transform(self, X):
        """拟合并转换"""
        return self.scaler.fit_transform(X)

    def transform(self, X):
        """仅转换"""
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        """反向转换"""
        return self.scaler.inverse_transform(X)

# 使用示例
features = ['molecular_weight', 'temperature', 'shear_rate', 'pdi']
X = df_clean[features].values

normalizer = DataNormalizer(method='standard')
X_normalized = normalizer.fit_transform(X)

print(f"标准化前均值: {X.mean(axis=0)}")
print(f"标准化后均值: {X_normalized.mean(axis=0)}")
print(f"标准化后标准差: {X_normalized.std(axis=0)}")
```

---

## 3. 特征工程

### 3.1 基础特征构建

```python
class FeatureEngineer:
    """特征工程器"""

    @staticmethod
    def create_polynomial_features(df, columns, degree=2):
        """创建多项式特征"""
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[columns])
        feature_names = poly.get_feature_names_out(columns)

        df_poly = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
        return pd.concat([df, df_poly], axis=1)

    @staticmethod
    def create_interaction_features(df, feature_pairs):
        """创建交互特征"""
        for feat1, feat2 in feature_pairs:
            df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
            df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-8)
        return df

    @staticmethod
    def create_domain_features(df):
        """创建领域特征(流变学相关)"""
        # WLF方程相关
        if 'temperature' in df.columns:
            df['T_reciprocal'] = 1 / (df['temperature'] + 273.15)
            df['log_T'] = np.log(df['temperature'] + 273.15)

        # Arrhenius方程
        if 'temperature' in df.columns and 'viscosity' in df.columns:
            R = 8.314  # 气体常数
            df['ln_viscosity_T'] = np.log(df['viscosity']) / (R * (df['temperature'] + 273.15))

        # 分子量相关
        if 'molecular_weight' in df.columns:
            df['log_Mw'] = np.log10(df['molecular_weight'])

        # 剪切速率相关
        if 'shear_rate' in df.columns:
            df['log_shear_rate'] = np.log10(df['shear_rate'] + 1e-8)

        return df

# 使用示例
fe = FeatureEngineer()

# 创建多项式特征
df_featured = fe.create_polynomial_features(
    df_clean,
    columns=['temperature', 'shear_rate'],
    degree=2
)

# 创建交互特征
feature_pairs = [
    ('temperature', 'shear_rate'),
    ('molecular_weight', 'pdi')
]
df_featured = fe.create_interaction_features(df_featured, feature_pairs)

# 创建领域特征
df_featured = fe.create_domain_features(df_featured)

print(f"特征工程后的特征数: {df_featured.shape[1]}")
print(f"新增特征: {df_featured.columns.tolist()[-10:]}")
```

### 3.2 特征选择

```python
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor

class FeatureSelector:
    """特征选择器"""

    @staticmethod
    def select_by_correlation(df, target_col, threshold=0.1):
        """基于相关性选择特征"""
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        selected_features = correlations[correlations > threshold].index.tolist()
        selected_features.remove(target_col)
        print(f"选择了 {len(selected_features)} 个特征")
        return selected_features

    @staticmethod
    def select_by_univariate(X, y, k=10):
        """单变量特征选择"""
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        scores = pd.DataFrame({
            'feature': range(X.shape[1]),
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        print(scores.head(k))
        return selector.get_support(indices=True)

    @staticmethod
    def select_by_rfe(X, y, n_features=10):
        """递归特征消除"""
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector.fit(X, y)
        return selector.get_support(indices=True)

    @staticmethod
    def select_by_importance(X, y, feature_names, threshold=0.01):
        """基于特征重要性选择"""
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        selected = importances[importances['importance'] > threshold]['feature'].tolist()
        print(importances.head(10))
        return selected

# 使用示例
selector = FeatureSelector()

# 方法1: 相关性筛选
selected_features = selector.select_by_correlation(
    df_featured,
    target_col='viscosity',
    threshold=0.15
)

# 方法2: 特征重要性
X = df_featured[features].values
y = df_featured['viscosity'].values
selected_features = selector.select_by_importance(X, y, features, threshold=0.01)
```

---

## 4. 模型训练

### 4.1 传统机器学习模型

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

class MLModelTrainer:
    """机器学习模型训练器"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """准备训练和测试数据"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def train_linear_models(self, X_train, y_train):
        """训练线性模型"""
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            self.models[name] = model
            print(f"{name} 训练完成")

    def train_tree_models(self, X_train, y_train):
        """训练树模型"""
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            self.models[name] = model
            print(f"{name} 训练完成")

    def train_svr(self, X_train, y_train):
        """训练支持向量回归"""
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        model.fit(X_train, y_train)
        self.models['SVR'] = model
        print("SVR 训练完成")

    def evaluate_models(self, X_test, y_test):
        """评估所有模型"""
        results = []

        for name, model in self.models.items():
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape
            })

        results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
        self.results = results_df
        return results_df

# 使用示例
trainer = MLModelTrainer()

# 准备数据
X = df_featured[selected_features].values
y = df_featured['viscosity'].values
X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)

# 训练模型
trainer.train_linear_models(X_train, y_train)
trainer.train_tree_models(X_train, y_train)
trainer.train_svr(X_train, y_train)

# 评估模型
results = trainer.evaluate_models(X_test, y_test)
print("\n模型性能对比:")
print(results)
```

### 4.2 神经网络模型

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class RheologyNN(nn.Module):
    """流变性能预测神经网络"""

    def __init__(self, input_dim, hidden_dims=[64, 32, 16], dropout=0.2):
        super(RheologyNN, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class NNTrainer:
    """神经网络训练器"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': []}

    def train(self, train_loader, val_loader, epochs=100, lr=0.001):
        """训练模型"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证阶段
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            scheduler.step(val_loss)

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"早停于 epoch {epoch+1}")
                break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))

    def predict(self, X):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions

# 使用示例
# 准备数据
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_val_tensor = torch.FloatTensor(X_test)
y_val_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 创建和训练模型
input_dim = X_train.shape[1]
model = RheologyNN(input_dim, hidden_dims=[128, 64, 32], dropout=0.3)
trainer = NNTrainer(model)
trainer.train(train_loader, val_loader, epochs=200, lr=0.001)

# 预测
y_pred = trainer.predict(X_test)
print(f"神经网络 R2 Score: {r2_score(y_test, y_pred):.4f}")
```

---

## 5. 模型评估

### 5.1 评估指标

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """模型评估器"""

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """计算评估指标"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }

        return metrics

    @staticmethod
    def plot_predictions(y_true, y_pred, model_name='Model'):
        """绘制预测结果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 散点图
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()],
                     [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('True Values')
        axes[0].set_ylabel('Predictions')
        axes[0].set_title(f'{model_name} - Predictions vs True Values')

        # 残差图
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predictions')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title(f'{model_name} - Residual Plot')

        plt.tight_layout()
        plt.savefig(f'{model_name}_evaluation.png', dpi=300)
        plt.show()

    @staticmethod
    def cross_validate(model, X, y, cv=5):
        """交叉验证"""
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(model, X, y, cv=cv,
                                scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)

        print(f"交叉验证 RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")
        return rmse_scores

    @staticmethod
    def plot_learning_curve(train_sizes, train_scores, val_scores, model_name='Model'):
        """绘制学习曲线"""
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, val_mean, label='Validation score')
        plt.fill_between(train_sizes, train_mean - train_std,
                         train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, val_mean - val_std,
                         val_mean + val_std, alpha=0.1)
        plt.xlabel('Training Size')
        plt.ylabel('Score')
        plt.title(f'{model_name} - Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{model_name}_learning_curve.png', dpi=300)
        plt.show()

# 使用示例
evaluator = ModelEvaluator()

# 计算指标
metrics = evaluator.calculate_metrics(y_test, y_pred)
print("评估指标:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# 可视化
evaluator.plot_predictions(y_test, y_pred.flatten(), model_name='XGBoost')

# 交叉验证
best_model = trainer.models['XGBoost']
cv_scores = evaluator.cross_validate(best_model, X_train, y_train, cv=5)
```

---

## 6. 模型解释

### 6.1 SHAP分析

```python
import shap

class ModelInterpreter:
    """模型解释器"""

    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names

    def shap_analysis(self, X_test):
        """SHAP分析"""
        # 创建解释器
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)

        # 汇总图
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test,
                         feature_names=self.feature_names, show=False)
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300)
        plt.show()

        # 条形图
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test,
                         feature_names=self.feature_names,
                         plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig('shap_bar.png', dpi=300)
        plt.show()

        return shap_values

    def feature_importance(self):
        """特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 8))
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)),
                      [self.feature_names[i] for i in indices])
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300)
            plt.show()

            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

    def partial_dependence_plot(self, features):
        """偏依赖图"""
        from sklearn.inspection import PartialDependenceDisplay

        fig, ax = plt.subplots(figsize=(12, 4))
        PartialDependenceDisplay.from_estimator(
            self.model, self.X_train, features,
            feature_names=self.feature_names, ax=ax
        )
        plt.tight_layout()
        plt.savefig('partial_dependence.png', dpi=300)
        plt.show()

# 使用示例
interpreter = ModelInterpreter(
    model=trainer.models['XGBoost'],
    X_train=X_train,
    feature_names=selected_features
)

# SHAP分析
shap_values = interpreter.shap_analysis(X_test)

# 特征重要性
importance_df = interpreter.feature_importance()
print(importance_df.head(10))

# 偏依赖图
interpreter.partial_dependence_plot([0, 1, 2])  # 前3个特征
```

---

## 7. 优化算法

### 7.1 超参数优化

```python
import optuna
from sklearn.model_selection import cross_val_score

class HyperparameterOptimizer:
    """超参数优化器"""

    def __init__(self, X_train, y_train, cv=5):
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv

    def optimize_xgboost(self, n_trials=100):
        """优化XGBoost超参数"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            }

            model = xgb.XGBRegressor(**params, random_state=42)
            scores = cross_val_score(model, self.X_train, self.y_train,
                                    cv=self.cv, scoring='neg_mean_squared_error')
            return -scores.mean()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print("最佳参数:")
        print(study.best_params)
        print(f"最佳RMSE: {np.sqrt(study.best_value):.4f}")

        return study.best_params

    def optimize_rf(self, n_trials=100):
        """优化随机森林超参数"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            }

            model = RandomForestRegressor(**params, random_state=42)
            scores = cross_val_score(model, self.X_train, self.y_train,
                                    cv=self.cv, scoring='neg_mean_squared_error')
            return -scores.mean()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print("最佳参数:")
        print(study.best_params)
        print(f"最佳RMSE: {np.sqrt(study.best_value):.4f}")

        return study.best_params

# 使用示例
optimizer = HyperparameterOptimizer(X_train, y_train, cv=5)

# 优化XGBoost
best_xgb_params = optimizer.optimize_xgboost(n_trials=50)

# 使用最佳参数训练模型
best_model = xgb.XGBRegressor(**best_xgb_params, random_state=42)
best_model.fit(X_train, y_train)
```

### 7.2 配方优化

```python
from scipy.optimize import differential_evolution
from sko.GA import GA
from sko.PSO import PSO

class FormulationOptimizer:
    """配方优化器"""

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def objective_function(self, x, target_viscosity):
        """目标函数"""
        x_reshaped = x.reshape(1, -1)
        pred_viscosity = self.model.predict(x_reshaped)[0]
        return (pred_viscosity - target_viscosity) ** 2

    def optimize_with_de(self, bounds, target_viscosity):
        """使用差分进化算法优化"""
        result = differential_evolution(
            lambda x: self.objective_function(x, target_viscosity),
            bounds=bounds,
            maxiter=1000,
            popsize=15,
            tol=0.01
        )

        optimal_params = dict(zip(self.feature_names, result.x))
        predicted_viscosity = self.model.predict(result.x.reshape(1, -1))[0]

        print("优化结果 (差分进化):")
        print(f"目标粘度: {target_viscosity}")
        print(f"预测粘度: {predicted_viscosity:.4f}")
        print(f"误差: {abs(predicted_viscosity - target_viscosity):.4f}")
        print("\n最优参数:")
        for name, value in optimal_params.items():
            print(f"  {name}: {value:.4f}")

        return optimal_params

    def optimize_with_ga(self, bounds, target_viscosity):
        """使用遗传算法优化"""
        ga = GA(
            func=lambda x: self.objective_function(x, target_viscosity),
            n_dim=len(bounds),
            size_pop=50,
            max_iter=500,
            lb=[b[0] for b in bounds],
            ub=[b[1] for b in bounds]
        )

        best_x, best_y = ga.run()

        optimal_params = dict(zip(self.feature_names, best_x))
        predicted_viscosity = self.model.predict(best_x.reshape(1, -1))[0]

        print("优化结果 (遗传算法):")
        print(f"目标粘度: {target_viscosity}")
        print(f"预测粘度: {predicted_viscosity:.4f}")
        print(f"误差: {abs(predicted_viscosity - target_viscosity):.4f}")
        print("\n最优参数:")
        for name, value in optimal_params.items():
            print(f"  {name}: {value:.4f}")

        return optimal_params

    def optimize_with_pso(self, bounds, target_viscosity):
        """使用粒子群算法优化"""
        pso = PSO(
            func=lambda x: self.objective_function(x, target_viscosity),
            n_dim=len(bounds),
            pop=40,
            max_iter=500,
            lb=[b[0] for b in bounds],
            ub=[b[1] for b in bounds]
        )

        best_x, best_y = pso.run()

        optimal_params = dict(zip(self.feature_names, best_x))
        predicted_viscosity = self.model.predict(best_x.reshape(1, -1))[0]

        print("优化结果 (粒子群算法):")
        print(f"目标粘度: {target_viscosity}")
        print(f"预测粘度: {predicted_viscosity:.4f}")
        print(f"误差: {abs(predicted_viscosity - target_viscosity):.4f}")
        print("\n最优参数:")
        for name, value in optimal_params.items():
            print(f"  {name}: {value:.4f}")

        return optimal_params

# 使用示例
form_optimizer = FormulationOptimizer(best_model, selected_features)

# 定义参数边界
bounds = [
    (10000, 500000),    # molecular_weight
    (150, 250),         # temperature
    (0.1, 1000),        # shear_rate
    (1.0, 3.0)          # pdi
]

# 目标粘度
target_viscosity = 5000  # Pa·s

# 使用不同算法优化
optimal_params_de = form_optimizer.optimize_with_de(bounds, target_viscosity)
optimal_params_ga = form_optimizer.optimize_with_ga(bounds, target_viscosity)
optimal_params_pso = form_optimizer.optimize_with_pso(bounds, target_viscosity)
```

---

## 8. 完整案例

### 8.1 端到端流变性能预测

```python
# 完整的端到端案例
import warnings
warnings.filterwarnings('ignore')

class RheologyMLPipeline:
    """流变性能预测完整流程"""

    def __init__(self):
        self.data_loader = None
        self.cleaner = None
        self.normalizer = None
        self.fe = None
        self.trainer = None
        self.best_model = None

    def run_pipeline(self, data_path, target_col='viscosity'):
        """运行完整流程"""

        print("=" * 60)
        print("步骤 1: 数据加载")
        print("=" * 60)
        loader = RheologyDataLoader(data_path)
        df = loader.load_viscosity_data("data.csv")

        print("\n" + "=" * 60)
        print("步骤 2: 数据清洗")
        print("=" * 60)
        cleaner = DataCleaner()
        df = cleaner.handle_missing_values(df)
        df = cleaner.remove_outliers(df, target_col)
        cleaner.check_data_quality(df)

        print("\n" + "=" * 60)
        print("步骤 3: 特征工程")
        print("=" * 60)
        fe = FeatureEngineer()
        df = fe.create_domain_features(df)
        print(f"特征数量: {df.shape[1]}")

        print("\n" + "=" * 60)
        print("步骤 4: 特征选择")
        print("=" * 60)
        selector = FeatureSelector()
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].values
        y = df[target_col].values

        # 基于特征重要性选择
        selected_features = selector.select_by_importance(X, y, feature_cols)
        X = df[selected_features].values

        print("\n" + "=" * 60)
        print("步骤 5: 数据标准化")
        print("=" * 60)
        normalizer = DataNormalizer(method='standard')
        X = normalizer.fit_transform(X)

        print("\n" + "=" * 60)
        print("步骤 6: 模型训练")
        print("=" * 60)
        trainer = MLModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)

        trainer.train_linear_models(X_train, y_train)
        trainer.train_tree_models(X_train, y_train)

        print("\n" + "=" * 60)
        print("步骤 7: 模型评估")
        print("=" * 60)
        results = trainer.evaluate_models(X_test, y_test)
        print(results)

        # 选择最佳模型
        best_model_name = results.iloc[0]['Model']
        self.best_model = trainer.models[best_model_name]
        print(f"\n最佳模型: {best_model_name}")

        print("\n" + "=" * 60)
        print("步骤 8: 模型解释")
        print("=" * 60)
        interpreter = ModelInterpreter(self.best_model, X_train, selected_features)
        shap_values = interpreter.shap_analysis(X_test)
        importance_df = interpreter.feature_importance()

        print("\n" + "=" * 60)
        print("步骤 9: 超参数优化")
        print("=" * 60)
        optimizer = HyperparameterOptimizer(X_train, y_train)
        best_params = optimizer.optimize_xgboost(n_trials=50)

        # 使用最佳参数重新训练
        self.best_model = xgb.XGBRegressor(**best_params, random_state=42)
        self.best_model.fit(X_train, y_train)

        print("\n" + "=" * 60)
        print("步骤 10: 最终评估")
        print("=" * 60)
        y_pred = self.best_model.predict(X_test)
        evaluator = ModelEvaluator()
        final_metrics = evaluator.calculate_metrics(y_test, y_pred)

        print("最终模型性能:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")

        evaluator.plot_predictions(y_test, y_pred,
                                  model_name=f'Final_{best_model_name}')

        print("\n" + "=" * 60)
        print("流程完成!")
        print("=" * 60)

        return self.best_model, final_metrics

# 运行完整流程
pipeline = RheologyMLPipeline()
final_model, metrics = pipeline.run_pipeline("data/raw")
```

---

## 总结

本文档提供了从数据预处理到模型部署的完整代码框架。您可以根据具体的研究需求进行修改和扩展。

### 关键要点

1. **数据质量**: 确保数据清洗和预处理的质量
2. **特征工程**: 结合领域知识构建有意义的特征
3. **模型选择**: 尝试多种模型,选择最适合的
4. **超参数优化**: 使用自动化工具优化模型性能
5. **模型解释**: 使用SHAP等工具理解模型决策
6. **交叉验证**: 确保模型的泛化能力

### 后续工作

1. 将模型保存为可部署格式 (pickle, ONNX)
2. 构建Web界面进行在线预测
3. 持续收集数据优化模型
4. 发表论文和开源代码

祝您的博士研究顺利!
