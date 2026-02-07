# -*- coding: utf-8 -*-
"""
BSM2 故障数据集合并工具
======================

将 F00-F12 的 13 个故障数据集合并为一个统一的训练数据集。

输出文件:
    - bsm2_train_data.csv: 训练集 (80%)
    - bsm2_test_data.csv: 测试集 (20%)
    - bsm2_full_data.csv: 完整数据集

使用方法:
    python merge_dataset.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def merge_fault_datasets(
    input_dir: str = "bsm2_fault_dataset",
    output_dir: str = ".",
    train_ratio: float = 0.8,
    random_seed: int = 42,
    exclude_columns: list = None,
    start_day: float = 3.0,  # 从故障发生时刻开始取数据
    total_samples: int = None,  # 总样本数量限制，None表示不限制
):
    """
    合并所有故障数据集为统一的训练数据集。
    
    Args:
        input_dir: 输入目录，包含 F00-F12 的 CSV 文件
        output_dir: 输出目录
        train_ratio: 训练集比例
        random_seed: 随机种子
        exclude_columns: 要排除的列名列表
        start_day: 从第几天开始取数据（故障在第3天发生）
        total_samples: 总样本数量限制，None表示不限制
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 故障文件映射
    fault_files = {
        0: "F00_normal.csv",
        1: "F01_mu_a_reduced.csv",
        2: "F02_mu_h_reduced.csv",
        3: "F03_settling_velocity_reduced.csv",
        4: "F04_do_setpoint_step.csv",
        5: "F05_internal_recycle_drift.csv",
        6: "F06_rainy_influent.csv",
        7: "F07_stormy_influent.csv",
        8: "F08_do_sensor_drift.csv",
        9: "F09_do_sensor_offset.csv",
        10: "F10_do_sensor_failure.csv",
        11: "F11_was_flow_drift.csv",
        12: "F12_thickener_drop.csv",
    }
    
    # 默认排除的列（非特征列）
    if exclude_columns is None:
        exclude_columns = [
            'fault_id',      # 故障ID字符串（已有数值label）
            'in_SD1', 'in_SD2', 'in_SD3', 'in_XD4', 'in_XD5',  # 虚拟状态（始终为0）
            'eff_SD1', 'eff_SD2', 'eff_SD3', 'eff_XD4', 'eff_XD5',
            'r5_SD1', 'r5_SD2', 'r5_SD3', 'r5_XD4', 'r5_XD5',
        ]
    
    all_data = []
    
    # 计算每个类别的采样数量
    num_classes = len(fault_files)
    samples_per_class = total_samples // num_classes if total_samples else None
    
    print("=" * 60)
    print("BSM2 故障数据集合并工具")
    print("=" * 60)
    print(f"输入目录: {input_path.absolute()}")
    print(f"输出目录: {output_path.absolute()}")
    print(f"起始时间: Day {start_day}")
    if total_samples:
        print(f"目标总样本数: {total_samples} (每类 {samples_per_class})")
    print(f"训练集比例: {train_ratio * 100:.0f}%")
    print("-" * 60)
    
    np.random.seed(random_seed)
    
    for fault_num, filename in fault_files.items():
        filepath = input_path / filename
        
        if not filepath.exists():
            print(f"[警告] 文件不存在: {filepath}")
            continue
        
        # 读取数据
        df = pd.read_csv(filepath)
        
        # 筛选故障发生后的数据（从 start_day 开始）
        df_filtered = df[df['time_day'] >= start_day].copy()
        
        # 确保 label 列为整数
        df_filtered['label'] = fault_num
        
        original_rows = len(df)
        filtered_rows = len(df_filtered)
        
        # 采样：如果设置了限制且数据量超过限制，则均匀采样
        if samples_per_class and filtered_rows > samples_per_class:
            # 均匀间隔采样，保持时序特性
            step = filtered_rows / samples_per_class
            indices = [int(i * step) for i in range(samples_per_class)]
            df_sampled = df_filtered.iloc[indices].copy()
            sampled_rows = len(df_sampled)
            print(f"[F{fault_num:02d}] {filename}: {original_rows} -> {filtered_rows} -> {sampled_rows} 行")
            all_data.append(df_sampled)
        else:
            print(f"[F{fault_num:02d}] {filename}: {original_rows} -> {filtered_rows} 行")
            all_data.append(df_filtered)
    
    if not all_data:
        print("[错误] 没有找到任何数据文件！")
        return
    
    # 合并所有数据
    print("-" * 60)
    print("合并数据集...")
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # 删除排除的列
    cols_to_drop = [col for col in exclude_columns if col in merged_df.columns]
    if cols_to_drop:
        merged_df = merged_df.drop(columns=cols_to_drop)
        print(f"已删除列: {cols_to_drop}")
    
    # 数据统计
    print("-" * 60)
    print("数据统计:")
    print(f"  总样本数: {len(merged_df)}")
    print(f"  特征列数: {len(merged_df.columns) - 2}")  # 减去 time_day 和 label
    print(f"  类别分布:")
    label_counts = merged_df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = count / len(merged_df) * 100
        print(f"    类别 {label:2d}: {count:6d} 样本 ({pct:5.2f}%)")
    
    # 分割训练集和测试集（时序分割，避免数据泄露）
    print("-" * 60)
    print("分割数据集（时序分割）...")
    print("  每个故障类别：前80%时间 -> 训练集，后20%时间 -> 测试集")
    
    train_dfs = []
    test_dfs = []
    
    for label in sorted(merged_df['label'].unique()):
        # 获取该类别的数据，按时间排序
        label_df = merged_df[merged_df['label'] == label].sort_values('time_day')
        
        # 时序分割：前 train_ratio 作为训练，后面作为测试
        split_idx = int(len(label_df) * train_ratio)
        train_dfs.append(label_df.iloc[:split_idx])
        test_dfs.append(label_df.iloc[split_idx:])
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # 打乱训练集顺序（测试集保持原序）
    train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # 保存完整数据集
    full_path = output_path / "bsm2_full_data.csv"
    merged_df.to_csv(full_path, index=False)
    print(f"完整数据集已保存: {full_path} ({len(merged_df)} 行)")
    
    # 保存训练集
    train_path = output_path / "bsm2_train_data.csv"
    train_df.to_csv(train_path, index=False)
    print(f"训练集已保存: {train_path} ({len(train_df)} 行)")
    
    # 保存测试集
    test_path = output_path / "bsm2_test_data.csv"
    test_df.to_csv(test_path, index=False)
    print(f"测试集已保存: {test_path} ({len(test_df)} 行)")
    
    # 输出列信息
    print("-" * 60)
    feature_cols = [col for col in merged_df.columns if col not in ['label', 'time_day']]
    print("特征列 (共 {} 列):".format(len(feature_cols)))
    for i, col in enumerate(feature_cols):
        print(f"  {i:2d}. {col}")
    
    print("=" * 60)
    print("合并完成！")
    print("=" * 60)
    
    return merged_df, train_df, test_df


if __name__ == "__main__":
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    
    merge_fault_datasets(
        input_dir=script_dir / "bsm2_fault_dataset",
        output_dir=script_dir,
        train_ratio=0.8,
        random_seed=42,
        start_day=3.0,  # 从故障发生时刻（第3天）开始取数据
        total_samples=20000,  # 总样本数量限制
    )
