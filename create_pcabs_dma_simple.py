"""
创建PC/ABS合金的DMA数据集 - 纯Python版本
"""

import random
import math
import csv
from pathlib import Path

random.seed(42)

def create_pcabs_dma_dataset(n_samples=500):
    """创建PC/ABS合金的DMA数据集"""

    data = []
    headers = ['PC_content', 'ABS_content', 'temperature', 'frequency',
               'strain_amplitude', 'storage_modulus', 'loss_modulus',
               'tan_delta', 'Tg']

    for _ in range(n_samples):
        # 输入参数
        pc_content = random.uniform(10, 90)
        abs_content = 100 - pc_content
        temperature = random.uniform(20, 180)
        frequency = random.choice([0.1, 1, 10, 100])
        strain_amplitude = random.uniform(0.01, 1.0)

        # 计算Tg
        Tg_PC = 150
        Tg_ABS = 105
        w_pc = pc_content / 100
        w_abs = abs_content / 100
        Tg_blend = 1 / (w_pc/Tg_PC + w_abs/Tg_ABS)
        Tg_blend *= random.gauss(1.0, 0.05)

        # 计算储能模量
        E_base_PC = 2300
        E_base_ABS = 2000
        E_base = w_pc * E_base_PC + w_abs * E_base_ABS

        if temperature < Tg_blend - 20:
            temp_factor = 1.0 - 0.002 * (temperature - 20)
        elif temperature > Tg_blend + 20:
            temp_factor = 0.05 + 0.02 * math.exp(-(temperature - Tg_blend)/30)
        else:
            temp_factor = 0.5 * (1 - math.tanh((temperature - Tg_blend)/10))

        freq_factor = 1 + 0.1 * math.log10(frequency) if frequency > 0 else 1
        E_storage = E_base * temp_factor * freq_factor

        # 计算损耗模量
        delta_T = abs(temperature - Tg_blend)
        loss_peak = 500 * math.exp(-(delta_T**2) / (2 * 15**2))
        baseline_loss = 0.02 * E_storage
        E_loss = loss_peak + baseline_loss

        # 计算tan δ
        tan_delta = E_loss / (E_storage + 1e-6)

        # 添加噪声
        E_storage *= random.gauss(1.0, 0.05)
        E_loss *= abs(random.gauss(1.0, 0.08))
        tan_delta = E_loss / (E_storage + 1e-6)

        data.append([
            round(pc_content, 2),
            round(abs_content, 2),
            round(temperature, 2),
            frequency,
            round(strain_amplitude, 4),
            round(max(E_storage, 10), 2),
            round(max(E_loss, 1), 2),
            round(max(tan_delta, 0.001), 4),
            round(Tg_blend, 2)
        ])

    return headers, data


if __name__ == "__main__":
    print("=" * 70)
    print("Creating PC/ABS DMA Dataset")
    print("=" * 70)

    # 创建数据
    headers, data = create_pcabs_dma_dataset(n_samples=500)

    # 保存数据
    output_path = 'polymer_rheology_ml/data/raw/pcabs_dma_data.csv'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)

    print(f"\n✅ Dataset created with {len(data)} samples")
    print(f"📊 Saved to: {output_path}")

    # 显示前几行
    print("\n" + "=" * 70)
    print("Sample Data (first 10 rows)")
    print("=" * 70)
    print(",".join(headers))
    for row in data[:10]:
        print(",".join(str(x) for x in row))

    # 统计信息
    print("\n" + "=" * 70)
    print("Dataset Statistics")
    print("=" * 70)

    # 计算每列的统计信息
    for i, header in enumerate(headers):
        values = [row[i] for row in data]
        if isinstance(values[0], (int, float)):
            print(f"\n{header}:")
            print(f"  Min: {min(values):.2f}")
            print(f"  Max: {max(values):.2f}")
            print(f"  Mean: {sum(values)/len(values):.2f}")

    print("\n" + "=" * 70)
    print("✅ PC/ABS DMA dataset creation completed!")
    print(f"📁 Next step: Run the analysis pipeline")
    print(f"   cd polymer_rheology_ml")
    print(f"   Update config/config.yaml to use this data")
    print("=" * 70)
