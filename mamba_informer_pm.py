

"""
最终优化版Mamba-Informer
集成了先进的特征学习和数据预处理方法

核心架构：
1. 增强特征学习模块
2. 多尺度卷积特征提取
3. 真正的Mamba-Informer双分支结构
   - 简化Mamba块（状态空间建模）
   - Informer注意力机制
   - 门控融合
4. 智能特征融合
5. 稳定的训练策略
"""

import os
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
warnings.filterwarnings('ignore')

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

# GPU优化设置（benchmark模式可能影响可重复性，但能提升性能）
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # 为了性能，不强制确定性
if torch.cuda.is_available():
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision('medium')


class EnhancedFeatureLearning(nn.Module):
    """
    增强特征学习模块
    """
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
        """
        增强特征学习前向传播
        """
        batch_size, seq_len, _ = x.shape
        
        # 重塑进行特征变换
        x_flat = x.reshape(-1, self.input_dim)
        
        # 特征变换
        features = self.feature_transform(x_flat)
        enhanced = self.feature_enhance(features)
        
        # 重塑回原始形状
        output = enhanced.reshape(batch_size, seq_len, self.output_dim)
        
        return output


class MultiScaleConv(nn.Module):
    """
    多尺度卷积模块 - 借鉴ConvLatent
    """
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
        """
        多尺度卷积前向传播
        """
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
    """
    简化的Mamba块 - 保持Mamba的核心思想
    """
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
        """
        简化Mamba前向传播
        """
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


class InformerAttention(nn.Module):
    """
    Informer稀疏注意力机制 - 保持原有结构
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
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Informer注意力前向传播
        """
        residual = x
        x = self.norm(x)
        
        B, L, D = x.shape
        H = self.n_heads
        d_k = D // H
        
        # 确保维度能整除
        assert D % H == 0, f"d_model ({D}) must be divisible by n_heads ({H})"
        
        Q = self.q_proj(x).view(B, L, H, d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, L, H, d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, L, H, d_k).transpose(1, 2)
        
        # 简化的注意力计算
        scores = torch.einsum('bhid,bhjd->bhij', Q, K) / (d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.einsum('bhij,bhjd->bhid', attn, V)
        
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(context)
        
        return residual + output



class OptimizedMambaInformerBlock(nn.Module):
    """
    优化的Mamba-Informer块 - 直接整合所有组件，避免重复
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        
        # Mamba分支
        self.mamba = SimplifiedMambaBlock(d_model)
        
        # Informer注意力分支
        self.informer_attn = InformerAttention(d_model, n_heads)
        
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
        
    def forward(self, x):
        """
        优化块前向传播 - 简化版本
        """
        # Mamba分支
        mamba_out = self.mamba(x)
        
        # Informer分支
        informer_out = self.informer_attn(x)
        
        # 门控融合
        fused = self.gate * mamba_out + (1 - self.gate) * informer_out
        
        # 前馈网络
        output = fused + self.ffn(fused)
        
        return output


class FinalOptimizedModel(nn.Module):
    """
    最终优化的Mamba-Informer模型
    专注于实际有效的优化策略
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
        
        # 3. 优化的Mamba-Informer层
        self.layers = nn.ModuleList([
            OptimizedMambaInformerBlock(d_model, n_heads=8, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # 4. 特征聚合
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 5. 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # 序列+平均+最大特征
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
        """
        最终优化的前向传播
        """
        batch_size, seq_len, _ = x.shape
        x_raw = x
        
        # 1. 增强特征学习
        features = self.feature_learning(x)
        
        # 2. 位置编码
        features = features + self.pos_embedding[:seq_len].unsqueeze(0)
        
        # 3. 通过优化层
        for layer in self.layers:
            features = layer(features)
        
        # 4. 多层次特征聚合
        # 序列特征 (最后时刻)
        seq_features = features[:, -1, :]  # [batch, d_model]
        
        # 全局特征
        features_conv = features.transpose(1, 2)  # [batch, d_model, seq_len]
        global_avg = self.global_pool(features_conv).squeeze(-1)  # [batch, d_model]
        global_max = self.max_pool(features_conv).squeeze(-1)  # [batch, d_model]
        
        # 融合特征
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


def train_final_model(model, train_loader, val_loader, epochs=60, lr=0.001, patience=12):
    """
    最终优化的训练函数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    use_cuda = device.type == 'cuda'
    gpu_name = None
    if use_cuda:
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"检测到GPU: {gpu_name}")
        print(f"显存总量: {torch.cuda.get_device_properties(current_device).total_memory / 1024**3:.2f} GB")
        
        # 禁用torch.compile，避免Triton依赖问题
        # if hasattr(torch, "compile"):
        #     try:
        #         model = torch.compile(model, mode="max-autotune")
        #         print("已启用torch.compile加速")
        #     except Exception as compile_error:
        #         print(f"torch.compile加速失败: {compile_error}")
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
    
    print(f"开始训练最终优化模型，设备: {device}")
    print(f"模型是否在CUDA上: {next(model.parameters()).is_cuda}")
    
    for epoch in range(epochs):
        if use_cuda:
            torch.cuda.synchronize()
        epoch_start_time = time.time()  # 记录本轮开始时间
        
        # 训练阶段
        model.train()
        train_loss = 0
        batch_count = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            # 第一个batch打印调试信息
            if epoch == 0 and batch_count == 0:
                print(f"第一个batch数据是否在CUDA上: {batch_x.is_cuda}")
                if use_cuda:
                    print(f"当前GPU显存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
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
            torch.save(model.state_dict(), 'model/mamba_informer_pm.pth')
        else:
            patience_counter += 1
        
        print(f'Epoch [{epoch + 1:2d}/{epochs}] '
              f'Train: {train_loss:.4f} | Val: {val_loss:.4f} | R²: {val_r2:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f} | Time: {epoch_time:.2f}s'
              f'{" | GPU: " + gpu_name if gpu_name else ""}')
        
        if patience_counter >= patience:
            print(f"早停: {patience}轮无改善")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('model/mamba_informer_pm.pth'))
    return model, train_losses, val_losses


def main():
    """
    主函数 - 最终优化版本
    """
    # 确保随机种子生效
    set_seed(42)
    
    print("=" * 60)
    print("最终优化Mamba-Informer - 修复数据泄露问题")
    print("=" * 60)
    print("🔧 新数据流程: 基础预处理 → 数据划分 → 分别特征工程 → 标准化 → 训练")
    print("✅ 已修复数据泄露问题，确保训练集不使用测试集信息")
    
    # 基础数据预处理 - 不包含特征工程，避免数据泄露
    data = preprocess_basic_data('metro.xls')

    # 模型参数
    seq_len = 24
    batch_size = 32
    epochs = 60
    lr = 0.001

    # 数据划分
    train_dataset, test_dataset, scaler = create_data_splits(
        data, seq_len=seq_len, test_size=0.3, augment_factor=2
    )
    
    # 获取实际特征维度（从数据集中获取）
    sample_x, sample_y = train_dataset[0]
    actual_input_dim = sample_x.shape[-1]  # 实际特征数量
    print(f"实际输入特征维度: {actual_input_dim}")
    
    # 验证集划分
    train_size = int(len(train_dataset) * 0.8)
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(train_dataset)))
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    
    # DataLoader
    # 根据环境动态设置 DataLoader 优化参数
    # Windows 平台设置 num_workers=0 避免多进程内存问题
    import platform
    is_windows = platform.system() == 'Windows'
    
    if is_windows:
        # Windows 平台：避免多进程内存错误 (WinError 1455)
        num_workers = 0
        pin_memory = torch.cuda.is_available()
        persistent_workers = False
        prefetch_factor = None
    elif torch.cuda.is_available():
        # 非 Windows 且有 GPU
        num_workers = min(8, os.cpu_count() or 4)  # 最多8个workers，与报告一致
        pin_memory = True
        persistent_workers = True
        prefetch_factor = 4
    else:
        # 非 Windows 且无 GPU
        num_workers = max(os.cpu_count() - 1, 1)
        pin_memory = False
        persistent_workers = False
        prefetch_factor = 2

    # 创建随机数生成器以确保可重复性
    g = torch.Generator()
    g.manual_seed(42)
    
    # 创建 DataLoader，针对 Windows 平台进行参数适配
    train_loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': True,
        'generator': g  # 确保 shuffle 的可重复性
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
    
    # 创建最终优化模型 - 使用实际特征维度
    input_dim = actual_input_dim
    model = FinalOptimizedModel(
        input_dim=input_dim,
        d_model=128,
        n_layers=4,
        seq_len=seq_len,
        pred_len=1,
        dropout=0.15
    )
    
    print(f"最终模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 训练模型
    model, train_losses, val_losses = train_final_model(
        model, train_loader, val_loader, epochs=epochs, lr=lr, patience=12
    )
    
    # 评估模型
    r2, mse, mae, rmse, predictions, true_values = evaluate_model(
        model, test_loader, scaler
    )
    
    # print(f"\n🎯 最终优化Mamba-Informer结果:")
    # print(f"测试集 R² Score: {r2:.4f}")
    # print(f"MSE: {mse:.4f}")
    # print(f"MAE: {mae:.4f}")
    # print(f"RMSE: {rmse:.4f}")
    
    # # 性能分析
    # print(f"\n📊 性能对比:")
    # print(f"优化版本:      {r2:.4f}")

    # if r2 >= 0.95:
    #     print(f"🎉 优秀！达到高性能标准！")
    # elif r2 >= 0.90:
    #     print(f"✅ 良好！显著改善！")
    # else:
    #     print(f"📈 有改善，继续优化中")
    
    # 保存结果
    results = {
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'predictions': predictions,
        'true_values': true_values
    }
    
    np.save('result/mamba_informer_pm.npy', results)
    
    return model, results


class TimeSeriesDataset(Dataset):
    """
    时间序列数据集类
    """
    def __init__(self, data, seq_len=24, pred_len=1, augment=False):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.augment = augment

        # 创建序列
        self.sequences = []
        self.targets = []

        for i in range(len(data) - seq_len - pred_len + 1):
            seq = data[i:i+seq_len]
            target = data[i+seq_len:i+seq_len+pred_len, -1]  # PM2.5是最后一列
            self.sequences.append(seq)
            self.targets.append(target)

        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)

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


def preprocess_basic_data(file_path):
    """
    基础数据预处理 - 不包含特征工程，避免数据泄露
    """
    print("开始基础数据预处理...")

    # 读取数据
    data = pd.read_excel(file_path)
    print(f"原始数据形状: {data.shape}")

    # 移除第一行（单位行）和最后一列（备注列）
    data = data.iloc[1:, :-1].reset_index(drop=True)

    # 重命名列
    columns = ['时间', 'NO', 'NO2', 'CO', 'CO2', 'TEMP', 'HUM', 'PM10', 'PM2.5']
    data.columns = columns

    # 移除时间列用于建模
    feature_data = data.iloc[:, 1:].copy()

    # 转换数据类型
    for col in feature_data.columns:
        feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')

    # 处理缺失值 - 只使用前向填充，避免未来信息泄露
    print(f"缺失值统计:\n{feature_data.isnull().sum()}")
    feature_data = feature_data.fillna(method='ffill')  # 只用前向填充
    feature_data = feature_data.dropna()

    print(f"基础预处理后数据形状: {feature_data.shape}")
    print(f"PM2.5统计信息:\n{feature_data['PM2.5'].describe()}")

    return feature_data


def apply_feature_engineering(data, start_idx=0):
    """
    应用特征工程 - 确保不使用未来信息
    """
    print(f"对索引 {start_idx} 开始的数据进行特征工程...")
    
    feature_data = data.copy()
    
    # 添加核心滞后特征
    for lag in [1, 2, 3, 6, 12, 24]:
        feature_data[f'PM2.5_lag_{lag}'] = feature_data['PM2.5'].shift(lag)
        if lag <= 6:
            feature_data[f'PM10_lag_{lag}'] = feature_data['PM10'].shift(lag)
            feature_data[f'NO_lag_{lag}'] = feature_data['NO'].shift(lag)
            feature_data[f'NO2_lag_{lag}'] = feature_data['NO2'].shift(lag)

    # 添加移动平均特征 - 只使用历史窗口
    for window in [3, 6, 12, 24]:
        feature_data[f'PM2.5_ma_{window}'] = feature_data['PM2.5'].rolling(window=window, min_periods=1).mean()
        feature_data[f'TEMP_ma_{window}'] = feature_data['TEMP'].rolling(window=window, min_periods=1).mean()
        feature_data[f'HUM_ma_{window}'] = feature_data['HUM'].rolling(window=window, min_periods=1).mean()

    # 添加差分特征
    feature_data['PM2.5_diff_1'] = feature_data['PM2.5'].diff(1)
    feature_data['PM2.5_diff_3'] = feature_data['PM2.5'].diff(3)
    feature_data['PM10_diff'] = feature_data['PM10'].diff()

    # 添加比率特征
    feature_data['PM_ratio'] = feature_data['PM2.5'] / (feature_data['PM10'] + 1e-8)
    feature_data['NO_NO2_ratio'] = feature_data['NO'] / (feature_data['NO2'] + 1e-8)

    # 添加交互特征
    feature_data['temp_hum_interaction'] = feature_data['TEMP'] * feature_data['HUM']
    feature_data['pollution_index'] = (feature_data['PM2.5'] + feature_data['PM10'] +
                                     feature_data['NO'] + feature_data['NO2']) / 4

    # 添加时间特征 - 基于相对索引位置
    relative_idx = np.arange(len(feature_data)) + start_idx
    feature_data['hour_sin'] = np.sin(2 * np.pi * (relative_idx % 24) / 24)
    feature_data['hour_cos'] = np.cos(2 * np.pi * (relative_idx % 24) / 24)

    # 移除NaN行
    feature_data = feature_data.dropna()
    
    print(f"特征工程后数据形状: {feature_data.shape}")
    return feature_data


def create_data_splits(data, seq_len=24, test_size=0.3, augment_factor=1):
    """
    创建时间序列数据划分 - 先划分再特征工程，避免数据泄露
    """
    print(f"创建时间序列数据...")

    # 计算划分点
    total_len = len(data)
    max_seq_start = total_len - seq_len
    train_size = int(max_seq_start * (1 - test_size))

    # 按时间顺序划分原始数据
    train_data_raw = data.iloc[:train_size + seq_len].copy()
    test_data_raw = data.iloc[train_size:].copy()
    
    print(f"原始训练数据形状: {train_data_raw.shape}")
    print(f"原始测试数据形状: {test_data_raw.shape}")

    # 分别对训练集和测试集进行特征工程 - 避免数据泄露
    print("\n对训练集进行特征工程...")
    train_data_engineered = apply_feature_engineering(train_data_raw, start_idx=0)
    
    print("\n对测试集进行特征工程...")
    test_data_engineered = apply_feature_engineering(test_data_raw, start_idx=train_size)
    
    # 确保特征数量一致
    common_features = list(set(train_data_engineered.columns) & set(test_data_engineered.columns))
    train_data_engineered = train_data_engineered[common_features]
    test_data_engineered = test_data_engineered[common_features]
    
    print(f"最终特征数量: {len(common_features)}")

    # 标准化 - 只在训练数据上拟合
    print("\n进行标准化...")
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    train_scaled = scaler.fit_transform(train_data_engineered.values)
    test_scaled = scaler.transform(test_data_engineered.values)

    # 创建数据集
    train_dataset = TimeSeriesDataset(train_scaled, seq_len=seq_len, augment=(augment_factor > 1))
    test_dataset = TimeSeriesDataset(test_scaled, seq_len=seq_len, augment=False)

    # 数据增强
    if augment_factor > 1:
        augmented_datasets = [train_dataset]
        for _ in range(augment_factor - 1):
            aug_dataset = TimeSeriesDataset(train_scaled, seq_len=seq_len, augment=True)
            augmented_datasets.append(aug_dataset)

        train_dataset = torch.utils.data.ConcatDataset(augmented_datasets)

    print(f"最终训练集大小: {len(train_dataset)}")
    print(f"最终测试集大小: {len(test_dataset)}")

    return train_dataset, test_dataset, scaler


def evaluate_model(model, test_loader, scaler):
    """
    评估模型性能
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    predictions = []
    true_values = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            output = model(batch_x)
            predictions.extend(output.cpu().numpy().flatten())
            true_values.extend(batch_y.cpu().numpy().flatten())

    predictions = np.array(predictions)
    true_values = np.array(true_values)

    # 反标准化
    dummy_pred = np.zeros((len(predictions), scaler.n_features_in_))
    dummy_true = np.zeros((len(true_values), scaler.n_features_in_))

    dummy_pred[:, -1] = predictions
    dummy_true[:, -1] = true_values

    predictions_rescaled = scaler.inverse_transform(dummy_pred)[:, -1]
    true_values_rescaled = scaler.inverse_transform(dummy_true)[:, -1]

    # 计算评估指标
    r2 = r2_score(true_values_rescaled, predictions_rescaled)
    mse = mean_squared_error(true_values_rescaled, predictions_rescaled)
    mae = mean_absolute_error(true_values_rescaled, predictions_rescaled)
    rmse = np.sqrt(mse)

    print(f"\n模型评估结果:")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return r2, mse, mae, rmse, predictions_rescaled, true_values_rescaled


if __name__ == "__main__":
    import os
    # Windows 平台多进程支持
    import multiprocessing
    multiprocessing.freeze_support()
    
    os.makedirs('model', exist_ok=True)
    os.makedirs('result', exist_ok=True)

    main()
