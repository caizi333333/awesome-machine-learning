"""
PC/ABS DMA数据分析演示

目标：预测储能模量（storage_modulus）基于材料配方和测试条件
"""

import csv
import math
import random
from pathlib import Path
from collections import defaultdict

print("=" * 80)
print("PC/ABS DMA Data Analysis - Machine Learning Demo")
print("=" * 80)

# ==================== 1. 数据加载 ====================
print("\n【步骤 1】加载数据")
print("-" * 80)

data_path = 'polymer_rheology_ml/data/raw/pcabs_dma_data.csv'
data = []
with open(data_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 转换数值类型
        data.append({
            'PC_content': float(row['PC_content']),
            'ABS_content': float(row['ABS_content']),
            'temperature': float(row['temperature']),
            'frequency': float(row['frequency']),
            'strain_amplitude': float(row['strain_amplitude']),
            'storage_modulus': float(row['storage_modulus']),
            'loss_modulus': float(row['loss_modulus']),
            'tan_delta': float(row['tan_delta']),
            'Tg': float(row['Tg'])
        })

print(f"✅ 数据加载完成: {len(data)} 个样本")

# ==================== 2. 数据探索 ====================
print("\n【步骤 2】数据探索")
print("-" * 80)

# 统计分析
def calculate_stats(values):
    n = len(values)
    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values) / n
    std_val = math.sqrt(variance)
    return {
        'min': min(values),
        'max': max(values),
        'mean': mean_val,
        'std': std_val
    }

print("\n关键变量统计:")
print(f"{'变量':<20} {'最小值':>10} {'最大值':>10} {'平均值':>10} {'标准差':>10}")
print("-" * 70)

for key in ['PC_content', 'temperature', 'frequency', 'storage_modulus', 'Tg']:
    values = [row[key] for row in data]
    stats = calculate_stats(values)
    print(f"{key:<20} {stats['min']:>10.2f} {stats['max']:>10.2f} "
          f"{stats['mean']:>10.2f} {stats['std']:>10.2f}")

# 相关性分析
print("\n\n储能模量与各因素的相关性:")
print("-" * 80)

def correlation(x_values, y_values):
    """计算Pearson相关系数"""
    n = len(x_values)
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_values))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_values))

    if denominator_x == 0 or denominator_y == 0:
        return 0
    return numerator / (denominator_x * denominator_y)

storage_mod = [row['storage_modulus'] for row in data]
for key in ['PC_content', 'temperature', 'frequency', 'Tg']:
    values = [row[key] for row in data]
    corr = correlation(values, storage_mod)
    print(f"  {key:<20}: {corr:>7.4f}")

# ==================== 3. 特征工程 ====================
print("\n\n【步骤 3】特征工程")
print("-" * 80)

def create_features(row):
    """创建领域特征"""
    features = {
        # 原始特征
        'PC_content': row['PC_content'],
        'temperature': row['temperature'],
        'frequency': row['frequency'],
        'Tg': row['Tg'],

        # 领域特征
        'log_frequency': math.log10(row['frequency']) if row['frequency'] > 0 else 0,
        'T_minus_Tg': row['temperature'] - row['Tg'],  # 相对于Tg的温度
        'T_reciprocal': 1 / (row['temperature'] + 273.15),  # Arrhenius项
        'PC_squared': row['PC_content'] ** 2,

        # 交互特征
        'PC_x_temp': row['PC_content'] * row['temperature'],
        'freq_x_temp': row['frequency'] * row['temperature'],
    }
    return features

# 创建特征
X_data = [create_features(row) for row in data]
y_data = [row['storage_modulus'] for row in data]

feature_names = list(X_data[0].keys())
print(f"✅ 创建了 {len(feature_names)} 个特征")
print(f"特征列表: {', '.join(feature_names)}")

# ==================== 4. 数据分割 ====================
print("\n\n【步骤 4】数据分割")
print("-" * 80)

# 打乱数据
combined = list(zip(X_data, y_data))
random.seed(42)
random.shuffle(combined)
X_data, y_data = zip(*combined)
X_data, y_data = list(X_data), list(y_data)

# 80/20 分割
split_idx = int(len(X_data) * 0.8)
X_train, X_test = X_data[:split_idx], X_data[split_idx:]
y_train, y_test = y_data[:split_idx], y_data[split_idx:]

print(f"✅ 训练集: {len(X_train)} 样本")
print(f"✅ 测试集: {len(X_test)} 样本")

# ==================== 5. 简单线性回归模型 ====================
print("\n\n【步骤 5】模型训练 - 多元线性回归")
print("-" * 80)

def train_linear_regression(X, y):
    """简单的多元线性回归"""
    n = len(X)
    p = len(X[0])

    # 添加偏置项
    X_with_bias = []
    for x in X:
        X_with_bias.append([1.0] + [x[key] for key in feature_names])

    # 使用正规方程: w = (X^T X)^(-1) X^T y
    # 这里用简化的梯度下降

    # 初始化权重
    weights = [0.0] * (p + 1)

    # 梯度下降
    learning_rate = 1e-6
    iterations = 1000

    for iteration in range(iterations):
        # 计算预测
        predictions = []
        for x_row in X_with_bias:
            pred = sum(w * x for w, x in zip(weights, x_row))
            predictions.append(pred)

        # 计算梯度
        gradients = [0.0] * (p + 1)
        for i, (x_row, pred, true_y) in enumerate(zip(X_with_bias, predictions, y)):
            error = pred - true_y
            for j in range(len(weights)):
                gradients[j] += error * x_row[j]

        # 更新权重
        for j in range(len(weights)):
            weights[j] -= learning_rate * gradients[j] / n

        # 每100次迭代打印一次
        if (iteration + 1) % 200 == 0:
            mse = sum((pred - true_y) ** 2 for pred, true_y in zip(predictions, y)) / n
            rmse = math.sqrt(mse)
            print(f"  Iteration {iteration+1:4d}: RMSE = {rmse:.2f} MPa")

    return weights, feature_names

weights, features = train_linear_regression(X_train, y_train)

print("\n✅ 模型训练完成")
print(f"\n特征权重:")
print(f"{'特征':<20} {'权重':>15}")
print("-" * 40)
print(f"{'Bias (截距)':<20} {weights[0]:>15.2f}")
for i, feat in enumerate(features):
    print(f"{feat:<20} {weights[i+1]:>15.2f}")

# ==================== 6. 模型评估 ====================
print("\n\n【步骤 6】模型评估")
print("-" * 80)

def predict(X, weights, feature_names):
    """预测函数"""
    predictions = []
    for x in X:
        x_values = [1.0] + [x[key] for key in feature_names]
        pred = sum(w * val for w, val in zip(weights, x_values))
        predictions.append(pred)
    return predictions

def evaluate(y_true, y_pred):
    """计算评估指标"""
    n = len(y_true)

    # MSE
    mse = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n
    rmse = math.sqrt(mse)

    # MAE
    mae = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n

    # R²
    y_mean = sum(y_true) / n
    ss_tot = sum((yt - y_mean) ** 2 for yt in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # MAPE
    mape = sum(abs((yt - yp) / yt) for yt, yp in zip(y_true, y_pred) if yt != 0) / n * 100

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

# 训练集评估
y_train_pred = predict(X_train, weights, features)
train_metrics = evaluate(y_train, y_train_pred)

# 测试集评估
y_test_pred = predict(X_test, weights, features)
test_metrics = evaluate(y_test, y_test_pred)

print("训练集性能:")
print(f"  RMSE: {train_metrics['rmse']:.2f} MPa")
print(f"  MAE:  {train_metrics['mae']:.2f} MPa")
print(f"  R²:   {train_metrics['r2']:.4f}")
print(f"  MAPE: {train_metrics['mape']:.2f}%")

print("\n测试集性能:")
print(f"  RMSE: {test_metrics['rmse']:.2f} MPa")
print(f"  MAE:  {test_metrics['mae']:.2f} MPa")
print(f"  R²:   {test_metrics['r2']:.4f}")
print(f"  MAPE: {test_metrics['mape']:.2f}%")

# ==================== 7. 预测示例 ====================
print("\n\n【步骤 7】预测示例")
print("-" * 80)

# 创建几个示例预测
test_cases = [
    {'PC_content': 50, 'temperature': 25, 'frequency': 1, 'Tg': 125, 'description': 'PC/ABS 50/50, 室温, 1Hz'},
    {'PC_content': 70, 'temperature': 25, 'frequency': 1, 'Tg': 140, 'description': 'PC/ABS 70/30, 室温, 1Hz'},
    {'PC_content': 50, 'temperature': 100, 'frequency': 1, 'Tg': 125, 'description': 'PC/ABS 50/50, 100°C, 1Hz'},
    {'PC_content': 50, 'temperature': 150, 'frequency': 1, 'Tg': 125, 'description': 'PC/ABS 50/50, 150°C, 1Hz'},
]

header_text = "预测E' (MPa)"
print(f"\n{'配方和条件':<40} {header_text:>15}")
print("-" * 60)

for case in test_cases:
    features_dict = create_features(case)
    pred = predict([features_dict], weights, features)[0]
    print(f"{case['description']:<40} {pred:>15.2f}")

# ==================== 8. 关键发现 ====================
print("\n\n【步骤 8】关键发现")
print("=" * 80)

print("\n🔍 主要发现:")
print("  1. 温度是影响储能模量最重要的因素（负相关）")
print("  2. 在玻璃化转变温度(Tg)附近，模量急剧下降")
print("  3. PC含量影响Tg，进而影响储能模量")
print("  4. 频率对模量有正向影响（对数关系）")

print("\n📊 模型性能:")
print(f"  • 测试集R² = {test_metrics['r2']:.4f} (解释了{test_metrics['r2']*100:.1f}%的变异)")
print(f"  • 测试集RMSE = {test_metrics['rmse']:.2f} MPa")
print(f"  • 测试集MAPE = {test_metrics['mape']:.2f}%")

print("\n💡 工程应用:")
print("  • 可根据配方预测材料的动态力学性能")
print("  • 帮助优化PC/ABS配比以满足特定应用要求")
print("  • 减少实验次数，加速材料开发")

print("\n✅ 分析完成!")
print("=" * 80)

# ==================== 9. 保存结果 ====================
print("\n【步骤 9】保存预测结果")
print("-" * 80)

output_path = 'polymer_rheology_ml/data/processed/pcabs_predictions.csv'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['True_Storage_Modulus', 'Predicted_Storage_Modulus', 'Error', 'Absolute_Error'])

    for yt, yp in zip(y_test, y_test_pred):
        error = yp - yt
        abs_error = abs(error)
        writer.writerow([round(yt, 2), round(yp, 2), round(error, 2), round(abs_error, 2)])

print(f"✅ 预测结果已保存到: {output_path}")

print("\n" + "=" * 80)
print("🎉 PC/ABS DMA分析演示完成！")
print("=" * 80)
