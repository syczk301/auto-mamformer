"""
废水处理数据快速分析脚本
用于初步了解数据特征和BOD预测的可行性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib as mpl
import platform

# 根据操作系统选择合适的中文字体
system = platform.system()
if system == 'Windows':
    # Windows系统
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
elif system == 'Darwin':
    # macOS系统
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti']
else:
    # Linux系统
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

def load_and_overview(filepath):
    """加载数据并给出概览"""
    print("=" * 80)
    print("废水处理数据分析报告".center(80))
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv(filepath)
    
    print(f"\n1. 数据基本信息")
    print("-" * 80)
    print(f"   数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"   列名数量: {len(df.columns)}")
    print(f"\n   列名列表:")
    for i, col in enumerate(df.columns, 1):
        print(f"      {i:2d}. {col}")
    
    return df


def check_missing_values(df):
    """检查缺失值"""
    print(f"\n2. 缺失值分析")
    print("-" * 80)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    if missing.sum() == 0:
        print("   ✅ 数据完整，无缺失值")
    else:
        missing_df = pd.DataFrame({
            '缺失数量': missing[missing > 0],
            '缺失比例(%)': missing_pct[missing > 0]
        }).sort_values('缺失数量', ascending=False)
        print(missing_df)


def analyze_target_variable(df, target='BOD-S'):
    """分析目标变量"""
    print(f"\n3. 目标变量分析: {target}")
    print("-" * 80)
    
    if target not in df.columns:
        print(f"   ❌ 错误: 未找到目标变量 {target}")
        return None
    
    target_data = pd.to_numeric(df[target], errors='coerce')
    
    print(f"   统计信息:")
    print(f"      数量:     {target_data.count()}")
    print(f"      均值:     {target_data.mean():.2f} mg/L")
    print(f"      标准差:   {target_data.std():.2f} mg/L")
    print(f"      最小值:   {target_data.min():.2f} mg/L")
    print(f"      25分位:   {target_data.quantile(0.25):.2f} mg/L")
    print(f"      中位数:   {target_data.median():.2f} mg/L")
    print(f"      75分位:   {target_data.quantile(0.75):.2f} mg/L")
    print(f"      最大值:   {target_data.max():.2f} mg/L")
    print(f"      变异系数: {(target_data.std()/target_data.mean())*100:.1f}%")
    
    # 检查异常值（3σ原则）
    mean = target_data.mean()
    std = target_data.std()
    outliers = target_data[(target_data < mean - 3*std) | (target_data > mean + 3*std)]
    print(f"      异常值数: {len(outliers)} ({len(outliers)/len(target_data)*100:.1f}%)")
    
    return target_data


def correlation_analysis(df, target='BOD-S', top_n=15):
    """相关性分析"""
    print(f"\n4. 与{target}的相关性分析 (Top {top_n})")
    print("-" * 80)
    
    if target not in df.columns:
        print(f"   ❌ 错误: 未找到目标变量 {target}")
        return None
    
    # 转换所有列为数值型
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    
    # 计算相关系数
    correlations = df_numeric.corr()[target].sort_values(ascending=False)
    
    print(f"\n   正相关 Top {top_n//2}:")
    for feat, corr in correlations[1:top_n//2+1].items():
        print(f"      {feat:20s}: {corr:+.3f}")
    
    print(f"\n   负相关 Top {top_n//2}:")
    for feat, corr in correlations[-top_n//2:].items():
        print(f"      {feat:20s}: {corr:+.3f}")
    
    return correlations


def recommend_input_features(df, target='BOD-S', corr_threshold=0.1):
    """推荐输入特征"""
    print(f"\n5. 输入特征推荐")
    print("-" * 80)
    
    # 转换为数值型
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    
    # 计算相关性
    correlations = df_numeric.corr()[target].sort_values(ascending=False)
    
    # 基于相关性选择
    high_corr = correlations[abs(correlations) > corr_threshold].index.tolist()
    high_corr = [f for f in high_corr if f != target]
    
    print(f"   基于相关性 (|r| > {corr_threshold}):")
    print(f"      找到 {len(high_corr)} 个高相关特征")
    
    # 基于专业知识的推荐
    stages = {
        '进水阶段 (-E)': ['Q-E', 'BOD-E', 'COD-E', 'SS-E', 'VSS-E', 'PH-E', 'ZN-E'],
        '初沉池 (-P)': ['BOD-P', 'COD-P', 'SS-P', 'VSS-P', 'PH-P'],
        '曝气池 (-D)': ['BOD-D', 'COD-D', 'SS-D', 'VSS-D', 'PH-D'],
        '其他指标': ['RD-BOD-P', 'RD-SS-P', 'RD-COD-S']
    }
    
    print(f"\n   推荐特征分组:")
    recommended = []
    for stage, features in stages.items():
        available = [f for f in features if f in df.columns]
        if available:
            print(f"\n   【{stage}】")
            for feat in available:
                if feat in correlations.index:
                    corr = correlations[feat]
                    importance = "⭐⭐⭐" if abs(corr) > 0.3 else "⭐⭐" if abs(corr) > 0.15 else "⭐"
                    print(f"      {feat:15s}  相关性: {corr:+.3f}  {importance}")
                    if abs(corr) > corr_threshold:
                        recommended.append(feat)
                else:
                    print(f"      {feat:15s}  (无相关性数据)")
    
    print(f"\n   最终推荐: {len(recommended)} 个特征")
    print(f"   {recommended}")
    
    return recommended


def visualize_data(df, target='BOD-S', recommended_features=None):
    """可视化分析"""
    print(f"\n6. 生成可视化图表")
    print("-" * 80)
    
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    
    # 创建图表
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 目标变量分布
    ax1 = plt.subplot(2, 3, 1)
    target_data = df_numeric[target].dropna()
    ax1.hist(target_data, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(target_data.mean(), color='r', linestyle='--', label=f'均值: {target_data.mean():.1f}')
    ax1.axvline(target_data.median(), color='g', linestyle='--', label=f'中位数: {target_data.median():.1f}')
    ax1.set_xlabel(f'{target} (mg/L)')
    ax1.set_ylabel('频数')
    ax1.set_title(f'{target} 分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 目标变量时间序列
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(target_data.values, linewidth=0.8)
    ax2.set_xlabel('样本索引')
    ax2.set_ylabel(f'{target} (mg/L)')
    ax2.set_title(f'{target} 时间序列')
    ax2.grid(True, alpha=0.3)
    
    # 3. 箱线图
    ax3 = plt.subplot(2, 3, 3)
    box_data = [target_data]
    ax3.boxplot(box_data, labels=[target])
    ax3.set_ylabel('BOD-S (mg/L)')
    ax3.set_title(f'{target} 箱线图')
    ax3.grid(True, alpha=0.3)
    
    # 4. 相关性热力图 (top features)
    ax4 = plt.subplot(2, 3, 4)
    if recommended_features and len(recommended_features) > 0:
        plot_features = recommended_features[:8] + [target]
        corr_matrix = df_numeric[plot_features].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax4, cbar_kws={'label': '相关系数'})
        ax4.set_title('特征相关性热力图')
    else:
        ax4.text(0.5, 0.5, '无推荐特征', ha='center', va='center')
        ax4.set_title('相关性热力图')
    
    # 5. 前2个高相关特征与目标的散点图
    if recommended_features and len(recommended_features) >= 2:
        for i, feat in enumerate(recommended_features[:2]):
            ax = plt.subplot(2, 3, 5+i)
            valid_data = df_numeric[[feat, target]].dropna()
            ax.scatter(valid_data[feat], valid_data[target], alpha=0.5, s=10)
            
            # 计算趋势线
            z = np.polyfit(valid_data[feat], valid_data[target], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_data[feat].min(), valid_data[feat].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            corr = df_numeric[[feat, target]].corr().iloc[0, 1]
            ax.set_xlabel(feat)
            ax.set_ylabel(target)
            ax.set_title(f'{feat} vs {target}\n(r={corr:.3f})')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('result/wastewater_data_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ 图表已保存: result/wastewater_data_analysis.png")


def predictability_assessment(df, target='BOD-S'):
    """预测可行性评估"""
    print(f"\n7. 预测可行性评估")
    print("-" * 80)
    
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    
    # 数据量
    n_samples = len(df_numeric)
    score_data = 0
    if n_samples >= 1000:
        score_data = 5
        print(f"   数据量: {n_samples} 样本  ⭐⭐⭐⭐⭐ (充足)")
    elif n_samples >= 500:
        score_data = 4
        print(f"   数据量: {n_samples} 样本  ⭐⭐⭐⭐ (良好)")
    elif n_samples >= 300:
        score_data = 3
        print(f"   数据量: {n_samples} 样本  ⭐⭐⭐ (可用)")
    else:
        score_data = 2
        print(f"   数据量: {n_samples} 样本  ⭐⭐ (偏少，建议补充)")
    
    # 特征质量
    correlations = df_numeric.corr()[target].abs().sort_values(ascending=False)
    high_corr_count = (correlations > 0.3).sum() - 1  # 排除自己
    medium_corr_count = ((correlations > 0.1) & (correlations <= 0.3)).sum()
    
    score_feature = 0
    if high_corr_count >= 5:
        score_feature = 5
        print(f"   特征质量: {high_corr_count} 个强相关特征  ⭐⭐⭐⭐⭐ (优秀)")
    elif high_corr_count >= 3:
        score_feature = 4
        print(f"   特征质量: {high_corr_count} 个强相关特征  ⭐⭐⭐⭐ (良好)")
    elif high_corr_count >= 1:
        score_feature = 3
        print(f"   特征质量: {high_corr_count} 个强相关特征  ⭐⭐⭐ (可用)")
    else:
        score_feature = 2
        print(f"   特征质量: {high_corr_count} 个强相关特征  ⭐⭐ (需改进)")
    
    # 数据完整性
    missing_pct = df_numeric.isnull().sum().sum() / (len(df_numeric) * len(df_numeric.columns)) * 100
    score_quality = 0
    if missing_pct < 1:
        score_quality = 5
        print(f"   数据完整性: {100-missing_pct:.1f}%  ⭐⭐⭐⭐⭐ (优秀)")
    elif missing_pct < 5:
        score_quality = 4
        print(f"   数据完整性: {100-missing_pct:.1f}%  ⭐⭐⭐⭐ (良好)")
    elif missing_pct < 10:
        score_quality = 3
        print(f"   数据完整性: {100-missing_pct:.1f}%  ⭐⭐⭐ (可用)")
    else:
        score_quality = 2
        print(f"   数据完整性: {100-missing_pct:.1f}%  ⭐⭐ (需处理)")
    
    # 综合评分
    total_score = (score_data + score_feature + score_quality) / 3
    
    print(f"\n   综合评分: {total_score:.1f}/5.0")
    
    if total_score >= 4.5:
        print(f"   评估结论: ✅ 优秀 - 数据质量高，适合深度学习建模")
    elif total_score >= 3.5:
        print(f"   评估结论: ✅ 良好 - 数据可用，预期能取得不错效果")
    elif total_score >= 2.5:
        print(f"   评估结论: ⚠️  可用 - 建议进行数据增强或特征工程")
    else:
        print(f"   评估结论: ⚠️  需改进 - 建议补充数据或改进特征")


def main():
    """主函数"""
    import os
    os.makedirs('result', exist_ok=True)
    
    # 数据文件路径
    filepath = 'water-treatment_model_cleaned.csv'
    
    if not os.path.exists(filepath):
        print(f"❌ 错误: 未找到数据文件 {filepath}")
        return
    
    # 1. 加载数据
    df = load_and_overview(filepath)
    
    # 2. 缺失值分析
    check_missing_values(df)
    
    # 3. 目标变量分析
    target_data = analyze_target_variable(df, target='BOD-S')
    
    # 4. 相关性分析
    correlations = correlation_analysis(df, target='BOD-S', top_n=15)
    
    # 5. 推荐特征
    recommended = recommend_input_features(df, target='BOD-S', corr_threshold=0.1)
    
    # 6. 可视化
    visualize_data(df, target='BOD-S', recommended_features=recommended)
    
    # 7. 预测可行性评估
    predictability_assessment(df, target='BOD-S')
    
    print("\n" + "=" * 80)
    print("分析完成！".center(80))
    print("=" * 80)
    print("\n下一步:")
    print("  1. 查看生成的图表: result/wastewater_data_analysis.png")
    print("  2. 运行训练脚本: python mamba_informer_bod.py")
    print("\n")


if __name__ == "__main__":
    main()

