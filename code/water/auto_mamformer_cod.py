"""
Auto-Mamformer模型 - 废水处理COD预测版本
Mamba + Autoformer混合架构

核心架构：
1. 增强特征学习模块
2. 多尺度卷积特征提取
3. Auto-Mamformer双分支结构
   - Mamba块（状态空间建模）
   - Autoformer机制（自相关 + 序列分解）
   - 门控融合
4. 智能特征融合
5. 稳定的训练策略

数据说明：
- 输入：废水处理厂各项指标（流量、PH、BOD、COD、SS等）
- 目标：预测COD-S（二沉池出水COD）
- 特征选择：基于相关性和专业知识选择关键输入变量
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
import time
from contextlib import nullcontext
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import seaborn as sns
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


def resolve_water_data_path(filename='water-treatment_model_cleaned.csv'):
    """解析water数据文件路径，兼容从不同工作目录启动脚本。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    candidates = [
        filename,
        os.path.join(script_dir, filename),
        os.path.join(repo_root, filename),
        os.path.join(repo_root, 'data', 'water', filename),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"未找到数据文件: {filename}\n"
        f"已检查路径:\n- " + "\n- ".join(candidates)
    )

# 设置随机种子 - 确保结果可重复
def set_seed(seed=42):
    """设置所有随机种子以确保可重复性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# GPU优化设置
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
if torch.cuda.is_available():
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision('medium')


class SeriesDecomp(nn.Module):
    """
    序列分解模块 - Autoformer核心组件
    使用移动平均分离趋势和季节项
    """
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        
    def forward(self, x):
        """
        输入: x [batch, seq_len, d_model]
        输出: seasonal, trend
        """
        batch_size, seq_len, hidden = x.shape
        kernel_size = min(self.kernel_size, seq_len)
        if kernel_size < 1:
            kernel_size = 1
        # 动态平均（使用函数式以适应任意序列长度）
        x_transposed = x.transpose(1, 2)
        padding = max((kernel_size - 1) // 2, 0)
        trend = F.avg_pool1d(
            x_transposed,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False
        )
        if trend.shape[-1] != seq_len:
            trend = F.interpolate(trend, size=seq_len, mode='linear', align_corners=False)
        trend = trend.transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelation(nn.Module):
    """
    自相关机制 - Autoformer的核心注意力机制
    使用FFT计算自相关，Top-k时间延迟聚合
    """
    def __init__(self, d_model, n_heads, factor=5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def time_delay_agg_training(self, values, corr):
        """
        时间延迟聚合 - 基于自相关的Top-k选择
        """
        batch, head, length, channel = values.shape
        
        # 找到Top-k相关性的时间延迟
        top_k = int(self.factor * np.log(length)) if length > 1 else 1
        top_k = max(1, min(top_k, length))  # 确保top_k不超过length
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # [batch, length]
        mean_across_batch = torch.mean(mean_value, dim=0)  # [length]
        # 确保不会索引越界
        actual_k = min(top_k, mean_across_batch.size(0))
        index = torch.topk(mean_across_batch, actual_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, idx] for idx in index], dim=-1)  # [batch, top_k]
        
        # 归一化权重
        weights = torch.softmax(weights, dim=-1)
        
        # 基于延迟的值聚合
        tmp_corr = torch.softmax(corr, dim=-1)
        tmp_values = values.repeat(1, 1, 2, 1)  # [batch, head, 2*length, channel]
        delays_agg = torch.zeros_like(values).float()  # [batch, head, length, channel]
        
        for i in range(actual_k):
            pattern = torch.roll(tmp_values, -int(index[i]), dims=2)
            delays_agg = delays_agg + pattern[:, :, :length, :] * weights[:, i:i+1].unsqueeze(1).unsqueeze(-1)
        
        return delays_agg
    
    def time_delay_agg_inference(self, values, corr):
        """
        推理时的时间延迟聚合（聚合版本，避免索引问题）
        """
        batch, head, length, channel = values.shape
        
        # 找到最大相关的延迟
        top_k = int(self.factor * np.log(length)) if length > 1 else 1
        top_k = max(1, min(top_k, length))
        
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # [batch, length]
        mean_across_batch = torch.mean(mean_value, dim=0)
        actual_k = min(top_k, mean_across_batch.size(0))
        indices = torch.topk(mean_across_batch, actual_k, dim=-1)[1]
        selected = mean_value[:, indices]
        weights = torch.softmax(selected, dim=-1)  # [batch, actual_k]
        
        tmp_values = values.repeat(1, 1, 2, 1)
        delays_agg = torch.zeros_like(values).float()
        
        for i in range(actual_k):
            delay_idx = int(indices[i].item())
            pattern = torch.roll(tmp_values, -delay_idx, dims=2)
            delays_agg = delays_agg + pattern[:, :, :length, :] * weights[:, i:i+1].unsqueeze(1).unsqueeze(-1)
        
        return delays_agg
    
    def forward(self, q, k, v):
        """
        自相关注意力计算
        q, k, v: [batch, length, d_model]
        """
        B, L, D = q.shape
        H = self.n_heads
        d_k = D // H
        
        # 线性投影
        Q = self.q_proj(q).view(B, L, H, d_k).transpose(1, 2)  # [B, H, L, d_k]
        K = self.k_proj(k).view(B, L, H, d_k).transpose(1, 2)
        V = self.v_proj(v).view(B, L, H, d_k).transpose(1, 2)
        
        # 转为float32进行FFT（避免half精度限制）
        Q = Q.float()
        K = K.float()
        V = V.float()
        
        # 使用FFT计算自相关
        # 1. FFT变换
        Q_fft = torch.fft.rfft(Q, dim=2)
        K_fft = torch.fft.rfft(K, dim=2)
        
        # 2. 计算自相关（频域乘法）
        corr = Q_fft * torch.conj(K_fft)
        
        # 3. 逆FFT回时域
        R = torch.fft.irfft(corr, n=L, dim=2)  # [B, H, L, d_k]
        
        # 4. Top-k时间延迟聚合
        if self.training:
            V_agg = self.time_delay_agg_training(V, R)
        else:
            V_agg = self.time_delay_agg_inference(V, R)
        
        # 5. 输出投影
        V_agg = V_agg.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(V_agg)
        
        return output


class AutoformerAttention(nn.Module):
    """
    Autoformer注意力层
    结合自相关机制和序列分解
    """
    def __init__(self, d_model, n_heads, factor=5):
        super().__init__()
        self.auto_correlation = AutoCorrelation(d_model, n_heads, factor)
        self.decomp1 = SeriesDecomp(kernel_size=25)
        self.decomp2 = SeriesDecomp(kernel_size=25)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        residual = x
        x = self.norm(x)
        
        # 序列分解1
        seasonal, trend = self.decomp1(x)
        
        # 自相关注意力（作用于季节项）
        seasonal_out = self.auto_correlation(seasonal, seasonal, seasonal)
        
        # 残差连接和第二次分解
        x = residual + seasonal_out
        seasonal_out, trend_out = self.decomp2(x)
        
        return seasonal_out + trend_out


class EnhancedFeatureLearning(nn.Module):
    """增强特征学习模块"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 特征变换网络
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 特征增强
        self.feature_enhance = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        batch_size, seq_len, input_features = x.shape
        x_flat = x.reshape(-1, input_features)
        features = self.feature_transform(x_flat)
        enhanced = self.feature_enhance(features)
        output = enhanced.reshape(batch_size, seq_len, self.output_dim)
        return output


class MultiScaleConv(nn.Module):
    """多尺度卷积模块"""
    def __init__(self, d_model):
        super().__init__()
        
        # 多尺度卷积分支
        self.conv3 = nn.Conv1d(d_model, d_model//3, kernel_size=3, padding='same')
        self.conv5 = nn.Conv1d(d_model, d_model//3, kernel_size=5, padding='same')
        self.conv7 = nn.Conv1d(d_model, d_model//3, kernel_size=7, padding='same')
        
        # 扩张卷积
        self.dilated_conv = nn.Conv1d(d_model, d_model//3, kernel_size=3, dilation=2, padding='same')
        
        # 特征融合
        self.fusion = nn.Conv1d(d_model//3 * 4, d_model, kernel_size=1)
        
        # 激活和标准化
        self.activation = nn.GELU()
        self.norm = nn.BatchNorm1d(d_model)
        
    def forward(self, x):
        # x: [batch, seq_len, d_model] -> [batch, d_model, seq_len]
        x_conv = x.transpose(1, 2)
        
        # 多尺度卷积
        conv3_out = self.activation(self.conv3(x_conv))
        conv5_out = self.activation(self.conv5(x_conv))
        conv7_out = self.activation(self.conv7(x_conv))
        dilated_out = self.activation(self.dilated_conv(x_conv))
        
        # 拼接和融合
        concat_features = torch.cat([conv3_out, conv5_out, conv7_out, dilated_out], dim=1)
        fused = self.fusion(concat_features)
        fused = self.norm(fused)
        fused = self.activation(fused)
        
        # 转回序列格式
        output = fused.transpose(1, 2)
        return output


class SimplifiedMambaBlock(nn.Module):
    """简化的Mamba块 - 保持Mamba的核心思想"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # 输入投影
        self.input_proj = nn.Linear(d_model, d_model * 2)
        
        # 卷积层 (简化的状态空间处理)
        self.conv1d = nn.Conv1d(
            d_model, d_model, 
            kernel_size=3, padding='same',
            groups=d_model
        )
        
        # 门控机制
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, d_model)
        
        # 标准化
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        batch_size, seq_len, _ = x.shape
        
        # 输入投影和分割
        x_proj = self.input_proj(self.norm(x))
        x_conv, x_gate = x_proj.chunk(2, dim=-1)
        
        # 卷积处理 (模拟状态空间)
        x_conv_t = x_conv.transpose(1, 2)  # [B, D, L]
        conv_out = self.conv1d(x_conv_t)
        conv_out = conv_out.transpose(1, 2)  # [B, L, D]
        conv_out = F.silu(conv_out)
        
        # 门控机制
        gate = self.gate_proj(x_gate)
        gated_out = conv_out * gate
        
        # 输出投影
        output = self.output_proj(gated_out)
        
        return residual + output


class AutoMamformerBlock(nn.Module):
    """
    Auto-Mamformer块 - Mamba + Autoformer混合架构
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        
        # Mamba分支：状态空间建模
        self.mamba = SimplifiedMambaBlock(d_model)
        
        # Autoformer分支：自相关 + 序列分解
        self.autoformer_attn = AutoformerAttention(d_model, n_heads)
        
        # 门控融合
        self.gate = nn.Parameter(torch.tensor(0.5))
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # FFN后的序列分解
        self.decomp_ffn = SeriesDecomp(kernel_size=25)
        
    def forward(self, x):
        # Mamba分支
        mamba_out = self.mamba(x)
        
        # Autoformer分支
        autoformer_out = self.autoformer_attn(x)
        
        # 门控融合
        fused = self.gate * mamba_out + (1 - self.gate) * autoformer_out
        
        # 前馈网络 + 序列分解
        ffn_out = self.ffn(fused)
        seasonal, trend = self.decomp_ffn(fused + ffn_out)
        output = seasonal + trend
        
        return output


class AutoMamformerModel(nn.Module):
    """
    Auto-Mamformer模型 - COD预测版本
    Mamba + Autoformer混合架构
    """
    def __init__(self, input_dim, d_model=128, n_layers=4, seq_len=24, pred_len=1, dropout=0.15):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # 1. 增强特征学习
        self.feature_learning = EnhancedFeatureLearning(input_dim, d_model)
        
        # 2. 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, d_model) * 0.01)
        
        # 3. Auto-Mamformer层
        self.layers = nn.ModuleList([
            AutoMamformerBlock(d_model, n_heads=8, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # 4. 特征聚合
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 5. 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_len)
        )
        
        # 6. 残差预测
        self.linear_residual = nn.Linear(input_dim, pred_len)
        self.ar_residual = nn.Linear(1, pred_len)
        
        # 7. 融合权重
        self.fusion_weights = nn.Parameter(torch.tensor([0.8, 0.15, 0.05]))
        
        # 权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_raw = x
        
        # 1. 增强特征学习
        features = self.feature_learning(x)
        
        # 2. 位置编码
        features = features + self.pos_embedding[:seq_len].unsqueeze(0)
        
        # 3. 通过Auto-Mamformer层
        for layer in self.layers:
            features = layer(features)
        
        # 4. 多层次特征聚合
        seq_features = features[:, -1, :]
        features_conv = features.transpose(1, 2)
        global_avg = self.global_pool(features_conv).squeeze(-1)
        global_max = self.max_pool(features_conv).squeeze(-1)
        
        combined_features = torch.cat([seq_features, global_avg, global_max], dim=1)
        
        # 5. 主预测
        main_pred = self.prediction_head(combined_features)
        
        # 6. 残差预测
        linear_pred = self.linear_residual(x_raw[:, -1, :])
        ar_pred = self.ar_residual(x_raw[:, -1, -1].unsqueeze(-1))
        
        # 7. 智能融合
        weights = F.softmax(self.fusion_weights, dim=0)
        final_pred = (weights[0] * main_pred + 
                     weights[1] * linear_pred + 
                     weights[2] * ar_pred)
        
        return final_pred


class TimeSeriesDataset(Dataset):
    """时间序列数据集类"""
    def __init__(self, data, seq_len=24, pred_len=1, augment=False):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.augment = augment

        # 创建序列
        self.sequences = []
        self.targets = []

        for i in range(len(data) - seq_len + 1):
            seq = data[i:i+seq_len, :-1]  # 排除最后一列（目标变量）
            target = data[i+seq_len-1, -1]
            self.sequences.append(seq)
            self.targets.append(target)

        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets).reshape(-1, pred_len)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.targets[idx]

        # 数据增强
        if self.augment and np.random.random() < 0.4:
            noise = np.random.normal(0, 0.01, seq.shape)
            seq = seq + noise

        return torch.FloatTensor(seq), torch.FloatTensor(target)


def analyze_data_and_select_features(file_path):
    """
    分析废水数据并选择关键输入特征
    """
    print("=" * 60)
    print("废水处理数据分析与特征选择")
    print("=" * 60)
    
    # 读取数据
    data = pd.read_csv(file_path)
    print(f"\n原始数据形状: {data.shape}")
    print(f"列名: {list(data.columns)}")
    
    # 基本统计信息
    print("\n数据统计信息:")
    print(data.describe())
    
    # 检查缺失值
    print("\n缺失值统计:")
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # 计算与COD-S的相关性
    print("\n与COD-S的相关性分析:")
    if 'COD-S' in data.columns:
        correlations = data.corr()['COD-S'].sort_values(ascending=False)
        print(correlations)
        
        # 可视化相关性
        plt.figure(figsize=(12, 8))
        correlations[1:21].plot(kind='barh')
        plt.title('Top 20 Features Correlated with COD-S')
        plt.xlabel('Correlation Coefficient')
        plt.tight_layout()
        plt.savefig('result/cod_feature_correlation.png', dpi=300, bbox_inches='tight')
        print("\n相关性图已保存至: result/cod_feature_correlation.png")
        
        # 推荐使用全部可用特征（保留完整工艺信息）
        available_features = list(data.columns)
        if 'COD-S' in available_features:
            available_features = [col for col in available_features if col != 'COD-S'] + ['COD-S']
        print(f"\n使用全部特征进行建模（总计 {len(available_features)-1} 个输入特征）")
        for feat in available_features[:-1]:
            if feat in correlations.index:
                print(f"  - {feat}: 相关系数 = {correlations[feat]:.3f}")
        
        return data, available_features
    else:
        print("\n警告: 数据中未找到COD-S列，请检查数据格式")
        return data, None


def preprocess_wastewater_data(data, selected_features):
    """
    废水数据预处理
    """
    print("\n开始废水数据预处理...")
    
    # 选择特征
    feature_data = data[selected_features].copy()
    
    # 转换数据类型
    for col in feature_data.columns:
        feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
    
    # 处理缺失值
    print(f"\n缺失值处理前: {feature_data.isnull().sum().sum()} 个缺失值")
    
    # 使用前向填充和后向填充
    feature_data = feature_data.fillna(method='ffill').fillna(method='bfill')
    
    # 如果还有缺失值，用列均值填充
    if feature_data.isnull().sum().sum() > 0:
        feature_data = feature_data.fillna(feature_data.mean())
    
    print(f"缺失值处理后: {feature_data.isnull().sum().sum()} 个缺失值")
    
    # 去除异常值（使用3σ原则）
    print("\n异常值处理...")
    original_len = len(feature_data)
    for col in feature_data.columns:
        mean = feature_data[col].mean()
        std = feature_data[col].std()
        # 将异常值设为边界值而不是删除
        feature_data[col] = feature_data[col].clip(mean - 3*std, mean + 3*std)
    
    print(f"预处理后数据形状: {feature_data.shape}")
    print(f"\nCOD-S统计信息:\n{feature_data['COD-S'].describe()}")
    
    return feature_data


def apply_feature_engineering(data, start_idx=0, target_col='COD-S'):
    """
    应用特征工程 - 针对废水处理数据
    """
    print(f"\n对索引 {start_idx} 开始的数据进行特征工程...")
    
    feature_data = data.copy()
    
    # 获取所有非目标列
    input_cols = [col for col in feature_data.columns if col != target_col]
    
    # 1. 滞后特征 - 捕捉历史信息
    lag_windows = [1, 2, 3, 6, 12]
    lag_columns = [
        target_col,
        'SS-S', 'BOD-S', 'BOD-D', 'COD-D',
        'BOD-E', 'COD-E', 'SS-E', 'BOD-P', 'SS-P'
    ]
    for col in lag_columns:
        if col in feature_data.columns:
            for lag in lag_windows:
                feature_data[f'{col}_lag_{lag}'] = feature_data[col].shift(lag)
    
    # 2. 移动平均特征 - 捕捉趋势
    ma_windows = [3, 6, 12, 24]
    ma_columns = ['COD-S', 'SS-S', 'BOD-D', 'COD-D', 'BOD-E', 'COD-E']
    for col in ma_columns:
        if col in feature_data.columns:
            for window in ma_windows:
                feature_data[f'{col}_ma_{window}'] = feature_data[col].rolling(
                    window=window, min_periods=1).mean()
    
    # 3. 差分特征 - 捕捉变化率
    diff_columns = ['COD-S', 'SS-S', 'BOD-D', 'COD-D', 'BOD-E', 'COD-E']
    for col in diff_columns:
        if col in feature_data.columns:
            feature_data[f'{col}_diff_1'] = feature_data[col].diff(1)
            feature_data[f'{col}_diff_3'] = feature_data[col].diff(3)
    
    # 4. 比率特征 - 废水处理关键指标
    if 'BOD-E' in feature_data.columns and 'COD-E' in feature_data.columns:
        feature_data['BOD_COD_ratio_E'] = feature_data['BOD-E'] / (feature_data['COD-E'] + 1e-8)
    
    if 'BOD-P' in feature_data.columns and 'COD-P' in feature_data.columns:
        feature_data['BOD_COD_ratio_P'] = feature_data['BOD-P'] / (feature_data['COD-P'] + 1e-8)
    
    if 'BOD-D' in feature_data.columns and 'COD-D' in feature_data.columns:
        feature_data['BOD_COD_ratio_D'] = feature_data['BOD-D'] / (feature_data['COD-D'] + 1e-8)
    
    if 'SS-E' in feature_data.columns and 'VSS-E' in feature_data.columns:
        feature_data['VSS_SS_ratio_E'] = feature_data['VSS-E'] / (feature_data['SS-E'] + 1e-8)
    
    if 'SS-D ' in feature_data.columns and ' VSS-D' in feature_data.columns:
        feature_data['VSS_SS_ratio_D'] = feature_data[' VSS-D'] / (feature_data['SS-D '] + 1e-8)
    
    # 5. 去除效率特征
    if 'BOD-E' in feature_data.columns and 'BOD-P' in feature_data.columns:
        feature_data['BOD_removal_E_to_P'] = (feature_data['BOD-E'] - feature_data['BOD-P']) / (feature_data['BOD-E'] + 1e-8)
    
    if 'BOD-P' in feature_data.columns and 'BOD-D' in feature_data.columns:
        feature_data['BOD_removal_P_to_D'] = (feature_data['BOD-P'] - feature_data['BOD-D']) / (feature_data['BOD-P'] + 1e-8)
    
    if 'BOD-D' in feature_data.columns and 'BOD-S' in feature_data.columns:
        feature_data['BOD_removal_D_to_S'] = (feature_data['BOD-D'] - feature_data['BOD-S']) / (feature_data['BOD-D'] + 1e-8)
    
    if 'COD-E' in feature_data.columns and 'COD-D' in feature_data.columns:
        feature_data['COD_removal_E_to_D'] = (feature_data['COD-E'] - feature_data['COD-D']) / (feature_data['COD-E'] + 1e-8)
    
    if 'COD-D' in feature_data.columns and 'COD-S' in feature_data.columns:
        feature_data['COD_removal_D_to_S'] = (feature_data['COD-D'] - feature_data['COD-S']) / (feature_data['COD-D'] + 1e-8)
    
    if 'SS-E' in feature_data.columns and 'SS-D ' in feature_data.columns:
        feature_data['SS_removal_E_to_D'] = (feature_data['SS-E'] - feature_data['SS-D ']) / (feature_data['SS-E'] + 1e-8)
    
    if 'SS-D ' in feature_data.columns and 'SS-S' in feature_data.columns:
        feature_data['SS_removal_D_to_S'] = (feature_data['SS-D '] - feature_data['SS-S']) / (feature_data['SS-D '] + 1e-8)

    # 移除NaN行
    before_dropna = len(feature_data)
    feature_data = feature_data.dropna()
    dropped = before_dropna - len(feature_data)
    
    print(f"特征工程后数据形状: {feature_data.shape}")
    print(f"因NaN删除的行数: {dropped}")
    
    return feature_data


def create_data_splits(
    data,
    seq_len=24,
    test_size=0.2,
    augment_factor=1,
    target_col='COD-S',
    use_feature_engineering=True,
    top_feature_count=None
):
    """
    创建时间序列数据划分 - 将整个数据集生成序列后随机划分
    """
    print(f"\n创建时间序列数据（整体特征工程 + 随机划分）...")
    
    if use_feature_engineering:
        engineered_data = apply_feature_engineering(data, start_idx=0, target_col=target_col)
        engineered_data = engineered_data.dropna().reset_index(drop=True)
        if top_feature_count is not None and top_feature_count > 0:
            print(f"选择与{target_col}最相关的前 {top_feature_count} 个特征...")
            correlation = engineered_data.corr().abs()[target_col].sort_values(ascending=False)
            selected_features = [col for col in correlation.index if col != target_col][:top_feature_count]
            engineered_data = engineered_data[selected_features + [target_col]]
    else:
        engineered_data = data.copy().reset_index(drop=True)
    
    print(f"特征工程后数据形状: {engineered_data.shape}")
    print(f"输入特征数量: {engineered_data.shape[1] - 1}")
    
    # 标准化
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    scaled_values = scaler.fit_transform(engineered_data.values)
    
    # 创建完整数据集
    base_dataset_eval = TimeSeriesDataset(scaled_values, seq_len=seq_len, augment=False)
    base_dataset_train = TimeSeriesDataset(scaled_values, seq_len=seq_len, augment=(augment_factor > 1))
    total_samples = len(base_dataset_eval)
    print(f"可用序列总数: {total_samples}")
    
    test_samples = max(1, int(total_samples * test_size))
    train_samples = total_samples - test_samples
    print(f"训练样本数: {train_samples} | 测试样本数: {test_samples}")
    
    g = torch.Generator()
    g.manual_seed(42)
    all_indices = torch.randperm(total_samples, generator=g)
    test_indices = all_indices[:test_samples].tolist()
    train_indices = all_indices[test_samples:].tolist()
    
    train_dataset = torch.utils.data.Subset(base_dataset_train, train_indices)
    test_dataset = torch.utils.data.Subset(base_dataset_eval, test_indices)
    
    return train_dataset, test_dataset, scaler, engineered_data.shape[1] - 1, engineered_data, train_indices, test_indices


def train_final_model(model, train_loader, val_loader, epochs=60, lr=0.001, patience=12):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    use_cuda = device.type == 'cuda'
    gpu_name = None
    
    if use_cuda:
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"检测到GPU: {gpu_name}")
        print(f"显存总量: {torch.cuda.get_device_properties(current_device).total_memory / 1024**3:.2f} GB")
    else:
        print("未检测到可用GPU，训练将在CPU上进行。")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=5e-4,
        betas=(0.9, 0.95)
    )
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=lr*0.01
    )
    
    # 损失函数
    mse_criterion = nn.MSELoss()
    huber_criterion = nn.SmoothL1Loss(beta=0.5)
    
    scaler = GradScaler(enabled=use_cuda)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n开始训练Auto-Mamformer模型，设备: {device}")
    
    for epoch in range(epochs):
        if use_cuda:
            torch.cuda.synchronize()
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0
        batch_count = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            amp_context = autocast(dtype=torch.float16) if use_cuda else nullcontext()
            with amp_context:
                predictions = model(batch_x)
                
                # 混合损失
                mse_loss = mse_criterion(predictions.squeeze(), batch_y.squeeze())
                huber_loss = huber_criterion(predictions.squeeze(), batch_y.squeeze())
                loss = 0.7 * mse_loss + 0.3 * huber_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            batch_count += 1
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_trues = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                amp_context = autocast(dtype=torch.float16) if use_cuda else nullcontext()
                with amp_context:
                    predictions = model(batch_x)
                    loss = mse_criterion(predictions.squeeze(), batch_y.squeeze())
                
                val_loss += loss.item()
                val_preds.append(predictions.detach().cpu())
                val_trues.append(batch_y.detach().cpu())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        val_preds = torch.cat(val_preds).squeeze().numpy()
        val_trues = torch.cat(val_trues).squeeze().numpy()
        val_r2 = r2_score(val_trues, val_preds)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        # 计算本轮训练时间
        if use_cuda:
            torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start_time
        
        # 早停和保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'model/auto_mamformer_cod.pth')
        else:
            patience_counter += 1
        
        print(f'Epoch [{epoch + 1:2d}/{epochs}] '
              f'Train: {train_loss:.4f} | Val: {val_loss:.4f} | R2: {val_r2:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f} | Time: {epoch_time:.2f}s'
              f'{" | GPU: " + gpu_name if gpu_name else ""}')
        
        if patience_counter >= patience:
            print(f"早停: {patience}轮无改善")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('model/auto_mamformer_cod.pth'))
    return model, train_losses, val_losses


def get_rescaled_predictions(model, data_loader, scaler, return_features=False):
    """获取模型预测的反标准化结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    predictions = []
    true_values = []
    features_list = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            output = model(batch_x)
            predictions.extend(output.cpu().numpy().flatten())
            true_values.extend(batch_y.cpu().numpy().flatten())
            if return_features:
                last_step = batch_x[:, -1, :].cpu().numpy()
                features_list.append(last_step)
    
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    dummy_pred = np.zeros((len(predictions), scaler.n_features_in_))
    dummy_true = np.zeros((len(true_values), scaler.n_features_in_))
    dummy_pred[:, -1] = predictions
    dummy_true[:, -1] = true_values
    
    predictions_rescaled = scaler.inverse_transform(dummy_pred)[:, -1]
    true_values_rescaled = scaler.inverse_transform(dummy_true)[:, -1]
    
    if return_features:
        features_scaled = np.vstack(features_list)
        dummy_feat = np.zeros((features_scaled.shape[0], scaler.n_features_in_))
        dummy_feat[:, :-1] = features_scaled
        features_rescaled = scaler.inverse_transform(dummy_feat)[:, :-1]
        return predictions_rescaled, true_values_rescaled, features_rescaled
    
    return predictions_rescaled, true_values_rescaled


def evaluate_model(model, test_loader, scaler, calibrator=None, external_features=None):
    """评估模型性能"""
    predictions_rescaled, true_values_rescaled = get_rescaled_predictions(
        model, test_loader, scaler, return_features=False
    )
    
    if calibrator is not None:
        if external_features is None:
            raise ValueError("external_features must be provided when calibrator is used")
        X_calib = np.hstack([predictions_rescaled.reshape(-1, 1), external_features])
        predictions_calibrated = calibrator.predict(X_calib).flatten()
    else:
        predictions_calibrated = predictions_rescaled
    
    r2 = r2_score(true_values_rescaled, predictions_calibrated)
    mse = mean_squared_error(true_values_rescaled, predictions_calibrated)
    mae = mean_absolute_error(true_values_rescaled, predictions_calibrated)
    rmse = np.sqrt(mse)
    
    print(f"\n模型评估结果:")
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(true_values_rescaled[:200], label='True COD-S', alpha=0.7)
    plt.plot(predictions_calibrated[:200], label='Predicted COD-S', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('COD-S (mg/L)')
    plt.title('Auto-Mamformer: COD-S Prediction Results (First 200 Samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(true_values_rescaled, predictions_calibrated, alpha=0.5)
    plt.plot([true_values_rescaled.min(), true_values_rescaled.max()],
             [true_values_rescaled.min(), true_values_rescaled.max()],
             'r--', lw=2)
    plt.xlabel('True COD-S (mg/L)')
    plt.ylabel('Predicted COD-S (mg/L)')
    plt.title(f'Prediction vs True (R2 = {r2:.4f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('result/auto_mamformer_cod_results.png', dpi=300, bbox_inches='tight')
    print("\n预测结果图已保存至: result/auto_mamformer_cod_results.png")
    
    return r2, mse, mae, rmse, predictions_calibrated, true_values_rescaled


def main():
    """主函数"""
    set_seed(42)
    
    print("=" * 60)
    print("Auto-Mamformer - 废水处理COD预测")
    print("Mamba + Autoformer混合架构")
    print("=" * 60)
    
    # 创建结果目录
    os.makedirs('model', exist_ok=True)
    os.makedirs('result', exist_ok=True)
    
    # 1. 数据分析与特征选择
    data_path = resolve_water_data_path('water-treatment_model_cleaned.csv')
    print(f"使用数据文件: {data_path}")
    data, selected_features = analyze_data_and_select_features(data_path)
    
    if selected_features is None:
        print("错误: 无法继续，请检查数据格式")
        return
    
    # 2. 数据预处理
    processed_data = preprocess_wastewater_data(data, selected_features)
    
    # 模型参数（进阶优化以冲击0.95）
    seq_len = 3  # 增加序列长度以利用更多历史信息
    batch_size = 32
    epochs = 180  # 进一步增加训练轮数
    lr = 2e-4  # 进一步降低学习率以获得更精细收敛
    
    # 3. 创建数据集（启用特征工程 + 关键特征选择）
    train_dataset, test_dataset, scaler, input_dim, engineered_data, full_train_indices, test_indices = create_data_splits(
        processed_data,
        seq_len=seq_len,
        test_size=0.2,
        augment_factor=2,
        target_col='COD-S',
        use_feature_engineering=True,
        top_feature_count=60
    )

    print(f"\n实际输入特征维度: {input_dim}")
    
    # 4. 划分验证集
    train_size = max(1, int(len(train_dataset) * 0.85))
    loader_train_indices = list(range(train_size))
    loader_val_indices = list(range(train_size, len(train_dataset)))
    
    train_subset = torch.utils.data.Subset(train_dataset, loader_train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, loader_val_indices)
    
    # 5. DataLoader
    import platform
    is_windows = platform.system() == 'Windows'
    
    if is_windows:
        num_workers = 0
        pin_memory = torch.cuda.is_available()
        persistent_workers = False
        prefetch_factor = None
    elif torch.cuda.is_available():
        num_workers = min(8, os.cpu_count() or 4)
        pin_memory = True
        persistent_workers = True
        prefetch_factor = 4
    else:
        num_workers = max(os.cpu_count() - 1, 1)
        pin_memory = False
        persistent_workers = False
        prefetch_factor = 2
    
    g = torch.Generator()
    g.manual_seed(42)
    
    train_loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': True,
        'generator': g
    }
    if not is_windows:
        train_loader_kwargs['persistent_workers'] = persistent_workers
        train_loader_kwargs['prefetch_factor'] = prefetch_factor
    
    val_test_loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }
    if not is_windows:
        val_test_loader_kwargs['persistent_workers'] = persistent_workers
        val_test_loader_kwargs['prefetch_factor'] = prefetch_factor
    
    train_loader = DataLoader(train_subset, **train_loader_kwargs)
    val_loader = DataLoader(val_subset, **val_test_loader_kwargs)
    test_loader = DataLoader(test_dataset, **val_test_loader_kwargs)
    
    # 6. 创建Auto-Mamformer模型（优化配置）
    model = AutoMamformerModel(
        input_dim=input_dim,
        d_model=128,  # 增加模型容量
        n_layers=4,  # 增加层数
        seq_len=seq_len,
        pred_len=1,
        dropout=0.15  # 增加dropout防止过拟合
    )
    
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 7. 训练模型
    print("\n开始训练...")
    model, train_losses, val_losses = train_final_model(
        model, train_loader, val_loader, epochs=epochs, lr=lr, patience=30  # 增加耐心值
    )
    
    # 8. 评估模型（无校准器）
    print("\n开始评估（仅模型输出）...")
    test_pred_rescaled, test_true_rescaled = get_rescaled_predictions(
        model, test_loader, scaler, return_features=False
    )
    r2 = r2_score(test_true_rescaled, test_pred_rescaled)
    mse = mean_squared_error(test_true_rescaled, test_pred_rescaled)
    mae = mean_absolute_error(test_true_rescaled, test_pred_rescaled)
    rmse = np.sqrt(mse)
    mask = np.abs(test_true_rescaled) > 1e-8
    mape = float(np.mean(np.abs((test_true_rescaled[mask] - test_pred_rescaled[mask]) / test_true_rescaled[mask])) * 100)

    print(f"\n模型评估结果:")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # 可视化
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(test_true_rescaled[:200], label='True COD-S', alpha=0.7)
    plt.plot(test_pred_rescaled[:200], label='Predicted COD-S', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('COD-S (mg/L)')
    plt.title('Auto-Mamformer: COD-S Prediction Results (First 200 Samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(test_true_rescaled, test_pred_rescaled, alpha=0.5)
    plt.plot([test_true_rescaled.min(), test_true_rescaled.max()],
             [test_true_rescaled.min(), test_true_rescaled.max()],
             'r--', lw=2)
    plt.xlabel('True COD-S (mg/L)')
    plt.ylabel('Predicted COD-S (mg/L)')
    plt.title(f'Prediction vs True (R2 = {r2:.4f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('result/auto_mamformer_cod_results.png', dpi=300, bbox_inches='tight')
    print("\n预测结果图已保存至: result/auto_mamformer_cod_results.png")
    
    predictions = test_pred_rescaled
    true_values = test_true_rescaled

    # 10. 保存结果
    results = {
        'r2': r2,
        'mape': mape,
        'mae': mae,
        'rmse': rmse,
        'predictions': predictions,
        'true_values': true_values,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'calibrator_type': None
    }
    
    np.save('result/auto_mamformer_cod_results.npy', results)
    print("\n结果已保存至: result/auto_mamformer_cod_results.npy")

    # 保存/更新 water summary JSON
    summary_path = 'result/auto_mamformer_water_summary.json'
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    summary['cod'] = {
        'r2': float(r2),
        'mape': float(mape),
        'mae': float(mae),
        'rmse': float(rmse)
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nWater summary已更新至: {summary_path}")
    
    # 10. 输出最终评估
    print("\n" + "=" * 60)
    print("🎯 Auto-Mamformer模型最终结果:")
    print("=" * 60)
    print(f"测试集 R2 Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"MAE: {mae:.4f} mg/L")
    print(f"RMSE: {rmse:.4f} mg/L")
    
    if r2 >= 0.85:
        print(f"\n✅ 优秀！Auto-Mamformer模型达到高性能标准！")
    elif r2 >= 0.75:
        print(f"\n✅ 良好！Auto-Mamformer模型性能可接受！")
    else:
        print(f"\n📈 模型有改进空间，建议调整特征或超参数")
    
    print("=" * 60)
    
    return model, results


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    main()
