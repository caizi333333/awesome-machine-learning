"""
创建PC/ABS合金的DMA（动态力学分析）数据集

基于真实的物理规律和文献数据创建合成数据集
参考：PC/ABS合金的典型DMA性能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 设置随机种子
np.random.seed(42)

def create_pcabs_dma_dataset(n_samples=500):
    """
    创建PC/ABS合金的DMA数据集

    参数包括：
    - PC含量 (wt%)
    - ABS含量 (wt%)
    - 温度 (°C)
    - 频率 (Hz)
    - 应变幅值 (%)

    输出：
    - 储能模量 E' (MPa)
    - 损耗模量 E'' (MPa)
    - tan δ
    - 玻璃化转变温度 Tg (°C)
    """

    data = []

    for _ in range(n_samples):
        # 输入参数
        pc_content = np.random.uniform(10, 90)  # PC含量 10-90%
        abs_content = 100 - pc_content

        # 温度范围：室温到高温
        temperature = np.random.uniform(20, 180)  # °C

        # 频率：常见DMA测试频率
        frequency = np.random.choice([0.1, 1, 10, 100])  # Hz

        # 应变幅值
        strain_amplitude = np.random.uniform(0.01, 1.0)  # %

        # === 计算Tg（玻璃化转变温度）===
        # PC的Tg约150°C，ABS的Tg约105°C
        # 使用Fox方程预测共混物Tg
        Tg_PC = 150
        Tg_ABS = 105

        # Fox equation: 1/Tg = w1/Tg1 + w2/Tg2
        w_pc = pc_content / 100
        w_abs = abs_content / 100
        Tg_blend = 1 / (w_pc/Tg_PC + w_abs/Tg_ABS)

        # 添加一些偏差（相容性影响）
        compatibility_factor = np.random.normal(1.0, 0.05)
        Tg_blend *= compatibility_factor

        # === 计算储能模量 E' ===
        # E'在Tg以下较高，Tg以上急剧下降

        # 基础模量（室温下）
        # PC: ~2300 MPa, ABS: ~2000 MPa
        E_base_PC = 2300
        E_base_ABS = 2000
        E_base = w_pc * E_base_PC + w_abs * E_base_ABS

        # 温度对模量的影响（WLF-like behavior）
        if temperature < Tg_blend - 20:
            # 玻璃态
            temp_factor = 1.0 - 0.002 * (temperature - 20)
        elif temperature > Tg_blend + 20:
            # 橡胶态
            temp_factor = 0.05 + 0.02 * np.exp(-(temperature - Tg_blend)/30)
        else:
            # 转变区
            temp_factor = 0.5 * (1 - np.tanh((temperature - Tg_blend)/10))

        # 频率对模量的影响
        freq_factor = 1 + 0.1 * np.log10(frequency)

        E_storage = E_base * temp_factor * freq_factor

        # === 计算损耗模量 E'' ===
        # E''在Tg附近达到峰值

        # 计算到Tg的距离
        delta_T = abs(temperature - Tg_blend)

        # Gaussian-like peak at Tg
        loss_peak = 500 * np.exp(-(delta_T**2) / (2 * 15**2))

        # 基线损耗
        baseline_loss = 0.02 * E_storage

        E_loss = loss_peak + baseline_loss

        # === 计算 tan δ ===
        tan_delta = E_loss / (E_storage + 1e-6)

        # 添加噪声
        noise_factor = np.random.normal(1.0, 0.05)
        E_storage *= noise_factor
        E_loss *= abs(np.random.normal(1.0, 0.08))
        tan_delta = E_loss / (E_storage + 1e-6)

        data.append({
            # 输入特征
            'PC_content': pc_content,
            'ABS_content': abs_content,
            'temperature': temperature,
            'frequency': frequency,
            'strain_amplitude': strain_amplitude,

            # 输出性能
            'storage_modulus': max(E_storage, 10),  # MPa
            'loss_modulus': max(E_loss, 1),  # MPa
            'tan_delta': max(tan_delta, 0.001),
            'Tg': Tg_blend,
        })

    df = pd.DataFrame(data)
    return df


def plot_dma_curves(df, save_dir='polymer_rheology_ml/data/raw'):
    """绘制典型的DMA曲线"""

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 选择一个特定配比的样品进行展示
    pc_content = 50
    freq = 1

    sample_data = df[
        (df['PC_content'].between(pc_content-5, pc_content+5)) &
        (df['frequency'] == freq)
    ].sort_values('temperature')

    if len(sample_data) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # E' vs Temperature
        axes[0, 0].plot(sample_data['temperature'],
                       sample_data['storage_modulus'],
                       'b-', linewidth=2, label="E' (Storage Modulus)")
        axes[0, 0].set_xlabel('Temperature (°C)', fontsize=12)
        axes[0, 0].set_ylabel("E' (MPa)", fontsize=12)
        axes[0, 0].set_title(f'Storage Modulus - PC/ABS {pc_content}/{100-pc_content}', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # E'' vs Temperature
        axes[0, 1].plot(sample_data['temperature'],
                       sample_data['loss_modulus'],
                       'r-', linewidth=2, label="E'' (Loss Modulus)")
        axes[0, 1].set_xlabel('Temperature (°C)', fontsize=12)
        axes[0, 1].set_ylabel("E'' (MPa)", fontsize=12)
        axes[0, 1].set_title(f'Loss Modulus - PC/ABS {pc_content}/{100-pc_content}', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # tan δ vs Temperature
        axes[1, 0].plot(sample_data['temperature'],
                       sample_data['tan_delta'],
                       'g-', linewidth=2, label='tan δ')
        axes[1, 0].set_xlabel('Temperature (°C)', fontsize=12)
        axes[1, 0].set_ylabel('tan δ', fontsize=12)
        axes[1, 0].set_title(f'tan δ - PC/ABS {pc_content}/{100-pc_content}', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # Tg vs PC content
        tg_data = df.groupby(pd.cut(df['PC_content'], bins=10))['Tg'].mean()
        pc_bins = [interval.mid for interval in tg_data.index]
        axes[1, 1].plot(pc_bins, tg_data.values, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('PC Content (wt%)', fontsize=12)
        axes[1, 1].set_ylabel('Tg (°C)', fontsize=12)
        axes[1, 1].set_title('Glass Transition Temperature vs Composition', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/PCABS_DMA_curves.png', dpi=300, bbox_inches='tight')
        print(f"DMA curves saved to {save_dir}/PCABS_DMA_curves.png")
        plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("Creating PC/ABS DMA Dataset")
    print("=" * 70)

    # 创建数据集
    df = create_pcabs_dma_dataset(n_samples=500)

    # 保存数据
    output_path = 'polymer_rheology_ml/data/raw/pcabs_dma_data.csv'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nDataset created with {len(df)} samples")
    print(f"Saved to: {output_path}")

    print("\n" + "=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    print(df.describe())

    print("\n" + "=" * 70)
    print("Sample Data (first 5 rows)")
    print("=" * 70)
    print(df.head())

    print("\n" + "=" * 70)
    print("Data Info")
    print("=" * 70)
    print(df.info())

    # 绘制DMA曲线
    print("\n" + "=" * 70)
    print("Generating DMA Curves")
    print("=" * 70)
    plot_dma_curves(df)

    print("\n✅ PC/ABS DMA dataset creation completed!")
    print(f"📊 Dataset location: {output_path}")
    print(f"📈 Visualization: polymer_rheology_ml/data/raw/PCABS_DMA_curves.png")
