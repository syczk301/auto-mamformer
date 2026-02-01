import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from compare.lstm_model import LSTMRegressor
from compare.informer_model import InformerRegressor
from compare.mamba_model import MambaRegressor
from auto_mamformer_bod import set_seed


DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'water-treatment_model_cleaned.csv'))
TARGET_COL = 'BOD-S'
SEQ_LEN = 24
TEST_SIZE = 0.2
TOP_FEATURE_COUNT = 10
TRAIN_VAL_RATIO = 0.85  # 从训练集中再切 85/15 为 train/val
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
NUM_WORKERS = 0  # Windows 兼容


def inverse_scale_target(preds: np.ndarray, trues: np.ndarray, scaler):
    dummy_pred = np.zeros((len(preds), scaler.n_features_in_))
    dummy_true = np.zeros((len(trues), scaler.n_features_in_))
    dummy_pred[:, -1] = preds
    dummy_true[:, -1] = trues
    preds_rescaled = scaler.inverse_transform(dummy_pred)[:, -1]
    trues_rescaled = scaler.inverse_transform(dummy_true)[:, -1]
    return preds_rescaled, trues_rescaled


class SequenceDataset(Dataset):
    """简单序列数据集: (seq_len, features_except_target) -> target"""

    def __init__(self, arr: np.ndarray, seq_len: int):
        # arr: ndarray, 列最后一列为目标
        self.seq_len = seq_len
        self.arr = arr

    def __len__(self):
        return max(0, len(self.arr) - self.seq_len + 1)

    def __getitem__(self, idx):
        seq = self.arr[idx: idx + self.seq_len, :-1]
        target = self.arr[idx + self.seq_len - 1, -1]
        return torch.FloatTensor(seq), torch.FloatTensor([target])


def evaluate(model, loader, device, scaler):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            preds.append(out.squeeze(-1).cpu().numpy())
            trues.append(y.squeeze(-1).cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    preds_rescaled, trues_rescaled = inverse_scale_target(preds, trues, scaler)
    r2 = r2_score(trues_rescaled, preds_rescaled)
    mse = mean_squared_error(trues_rescaled, preds_rescaled)
    mae = mean_absolute_error(trues_rescaled, preds_rescaled)
    rmse = np.sqrt(mse)
    return {
        "r2": float(r2),
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "preds": preds_rescaled.tolist(),
        "trues": trues_rescaled.tolist(),
    }


def train_and_eval(model, train_loader, val_loader, test_loader, scaler, epochs=EPOCHS, lr=LR, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x).squeeze(-1)
            loss = criterion(out, y.squeeze(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                out = model(x).squeeze(-1)
                loss = criterion(out, y.squeeze(-1))
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1:02d} | Train {train_loss:.4f} | Val {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device, scaler)
    return model, test_metrics


def run_all():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 读取数据并数值化
    df = pd.read_csv(DATA_PATH)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if TARGET_COL not in df.columns:
        raise ValueError(f"目标列 {TARGET_COL} 不存在")
    df = df.ffill().bfill()

    # 时间顺序切分（避免序列重叠泄漏）
    total = len(df)
    test_rows = max(1, int(total * TEST_SIZE))
    train_rows = total - test_rows
    df_train = df.iloc[:train_rows].reset_index(drop=True)
    df_test = df.iloc[train_rows:].reset_index(drop=True)  # 测试集不与训练集重叠，避免泄漏

    # 特征工程：保持与主模型的核心一致性
    from auto_mamformer_bod import apply_feature_engineering
    engineered_train = apply_feature_engineering(df_train, start_idx=0, target_col=TARGET_COL)
    engineered_test = apply_feature_engineering(df_test, start_idx=0, target_col=TARGET_COL)

    # 过滤掉直接含有当前目标值的特征，只保留目标的滞后项；否则会泄露当期 y
    def filter_target_leak(df):
        keep = []
        for col in df.columns:
            if col == TARGET_COL:
                continue
            if col.startswith(f"{TARGET_COL}_lag_"):
                keep.append(col)
                continue
            if TARGET_COL in col:
                # 含有目标但不是滞后项，剔除
                continue
            keep.append(col)
        # 始终把目标列放在最后
        keep.append(TARGET_COL)
        return df[keep]

    engineered_train = filter_target_leak(engineered_train).reset_index(drop=True)
    engineered_test = filter_target_leak(engineered_test).reset_index(drop=True)

    # 选取与目标相关性最高的前 TOP_FEATURE_COUNT（基于过滤后的训练集）
    corr = engineered_train.corr().abs()[TARGET_COL].sort_values(ascending=False)
    available_feats = [col for col in corr.index if col != TARGET_COL]
    top_feats = available_feats[:min(TOP_FEATURE_COUNT, len(available_feats))]
    selected_features = top_feats + [TARGET_COL]
    engineered_train = engineered_train[selected_features].reset_index(drop=True)
    engineered_test = engineered_test[selected_features].reset_index(drop=True)

    # 训练集拟合 scaler，应用到 train/test
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    train_scaled = scaler.fit_transform(engineered_train.values)
    test_scaled = scaler.transform(engineered_test.values)

    # 构造序列数据集（顺序，不重叠泄漏）
    train_ds_full = SequenceDataset(train_scaled, seq_len=SEQ_LEN)
    test_ds = SequenceDataset(test_scaled, seq_len=SEQ_LEN)

    # 从训练集中再切 85/15 做 train/val（顺序切分）
    train_len = len(train_ds_full)
    train_size = max(1, int(train_len * TRAIN_VAL_RATIO))
    val_size = train_len - train_size
    indices = list(range(train_len))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = torch.utils.data.Subset(train_ds_full, train_indices)
    val_subset = torch.utils.data.Subset(train_ds_full, val_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    input_dim = len(selected_features) - 1

    configs = {
        "lstm": {"builder": lambda: LSTMRegressor(input_dim=input_dim, hidden_size=10, num_layers=1, dropout=0.65)},
        "informer": {"builder": lambda: InformerRegressor(input_dim=input_dim, d_model=20, n_layers=1, n_heads=2, dropout=0.6)},
        "mamba": {"builder": lambda: MambaRegressor(input_dim=input_dim, d_model=14, n_layers=1, dropout=0.7)},
    }

    results = {}
    for name, cfg in configs.items():
        print("\n" + "=" * 60)
        print(f"训练模型: {name}")
        model = cfg["builder"]()
        _, metrics = train_and_eval(
            model,
            train_loader,
            val_loader,
            test_loader,
            scaler,
            epochs=40,
            lr=1e-3,
            device=device,
        )
        results[name] = metrics
        print(f"{name} 测试集: R2={metrics['r2']:.4f} | RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f}")

    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'result'), exist_ok=True)
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'result', 'compare_results.json'))
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存到: {save_path}")


if __name__ == "__main__":
    run_all()

