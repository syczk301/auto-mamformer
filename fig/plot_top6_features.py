import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 字体设置，避免中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'water-treatment_model_cleaned.csv'))
TARGET_COL = 'BOD-S'
TOP_K = 6
SAMPLE_LEN = 400
XTICK_COUNT = 7
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def load_data():
    data = pd.read_csv(DATA_PATH)
    # 统一转为数值，无法转换的记为NaN
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    # 目标必须存在
    data = data.dropna(subset=[TARGET_COL])
    # 简单补全缺失
    data = data.ffill().bfill()
    return data


def select_top_features(data, top_k=TOP_K):
    corr = data.corr()[TARGET_COL].drop(TARGET_COL)
    corr_abs = corr.abs().sort_values(ascending=False)
    top_features = corr_abs.head(top_k).index.tolist()
    return top_features, corr.loc[top_features]


def plot_samples(data, features, out_path, sample_len=SAMPLE_LEN):
    sample_len = min(len(data), sample_len)
    x = np.arange(sample_len)
    rows, cols = 3, 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 9), sharex=False)
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        ax.plot(x, data[feat].values[:sample_len], label=feat, color=COLORS[i % len(COLORS)])
        ax.set_title(f"{feat}", fontsize=11)
        ax.set_ylabel("值")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=XTICK_COUNT, integer=True, prune=None))
        ax.grid(True, alpha=0.3)

    # 隐藏未使用子图
    for j in range(len(features), len(axes)):
        axes[j].axis('off')

    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"保存图片: {out_path}")


def main():
    data = load_data()
    top_features, corr = select_top_features(data, TOP_K)

    print("Top 6 原始特征及与 BOD-S 的相关系数：")
    for f in top_features:
        print(f"{f}: {corr[f]:.4f}")

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'result', 'top6_original_features.png'))
    plot_samples(data, top_features, out_path, SAMPLE_LEN)


if __name__ == "__main__":
    main()

