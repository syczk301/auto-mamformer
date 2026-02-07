"""
Auto-Mamformer on BSM2 dataset for COD and BOD5 prediction.

This keeps the same model architecture as `code/cod/auto_mamformer_cod.py`
and adapts only the BSM2 data/target construction and training entry.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")


# COD and BOD5 formula follows bsm2_python.bsm2.plantperformance.PlantPerformance.advanced_quantities
ASM1_FP = 0.08
COD_COMPONENTS = ("SI", "SS", "XI", "XS", "XBH", "XBA", "XP")
BOD_COMPONENTS = ("SS", "XS", "XBH", "XBA")


def load_auto_mamformer_components():
    """Load AutoMamformerModel and TimeSeriesDataset from existing COD script."""
    repo_root = Path(__file__).resolve().parents[2]
    source_file = repo_root / "code" / "cod" / "auto_mamformer_cod.py"
    if not source_file.exists():
        raise FileNotFoundError(f"Cannot find source model file: {source_file}")

    spec = importlib.util.spec_from_file_location("_auto_mamformer_cod_module", source_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for: {source_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AutoMamformerModel, module.TimeSeriesDataset, module.set_seed


def add_bsm2_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create COD/BOD5 targets for in/eff/r5 streams from ASM1 states."""
    df = df.copy()

    for prefix in ("in", "eff", "r5"):
        cod_cols = [f"{prefix}_{name}" for name in COD_COMPONENTS]
        bod_cols = [f"{prefix}_{name}" for name in BOD_COMPONENTS]

        if all(col in df.columns for col in cod_cols):
            df[f"{prefix}_COD"] = df[cod_cols].sum(axis=1)

        if all(col in df.columns for col in bod_cols):
            df[f"{prefix}_BOD5"] = 0.65 * (
                df[f"{prefix}_SS"]
                + df[f"{prefix}_XS"]
                + (1.0 - ASM1_FP) * (df[f"{prefix}_XBH"] + df[f"{prefix}_XBA"])
            )

    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Generic time-series feature engineering for BSM2 signals."""
    out = df.copy()

    trend_cols = [
        "in_Q",
        "in_COD",
        "in_BOD5",
        "r5_COD",
        "r5_BOD5",
        "r5_SNH",
        "r5_SNO",
        "r5_SO",
        "KLA4",
        "DO_sensor",
    ]
    trend_cols = [col for col in trend_cols if col in out.columns]

    for col in trend_cols:
        for lag in (1, 2, 3):
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
        out[f"{col}_ma3"] = out[col].rolling(window=3, min_periods=1).mean()
        out[f"{col}_ma6"] = out[col].rolling(window=6, min_periods=1).mean()
        out[f"{col}_diff1"] = out[col].diff(1)

    if "r5_SNO" in out.columns and "r5_SNH" in out.columns:
        out["r5_sno_snh_ratio"] = out["r5_SNO"] / (out["r5_SNH"] + 1e-8)

    if "in_COD" in out.columns and "in_SNH" in out.columns:
        out["in_cod_snh_ratio"] = out["in_COD"] / (out["in_SNH"] + 1e-8)

    if "r5_COD" in out.columns and "r5_SNH" in out.columns:
        out["r5_cod_snh_ratio"] = out["r5_COD"] / (out["r5_SNH"] + 1e-8)

    if "KLA4" in out.columns and "DO_sensor" in out.columns:
        out["kla4_do_interaction"] = out["KLA4"] * out["DO_sensor"]

    return out


def build_feature_candidates(df: pd.DataFrame) -> list[str]:
    """Select candidate feature columns while preventing direct effluent leakage."""
    excluded_exact = {"IQI", "EQI", "OCI", "eff_COD", "eff_BOD5"}
    feature_cols: list[str] = []

    for col in df.columns:
        if col in excluded_exact:
            continue
        if col.startswith("eff_"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        feature_cols.append(col)

    return feature_cols


def select_top_correlated_features(
    df: pd.DataFrame, target_col: str, candidates: list[str], top_k: int | None
) -> list[str]:
    """Pick top-k features by absolute Pearson correlation."""
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    usable = [col for col in candidates if col in df.columns]
    if not usable:
        raise ValueError("No usable candidate features.")

    corr_frame = df[usable + [target_col]].corr(numeric_only=True)
    corr = corr_frame[target_col].drop(labels=[target_col], errors="ignore").dropna()
    if corr.empty:
        return usable

    ranked = corr.abs().sort_values(ascending=False).index.tolist()
    if top_k is not None and top_k > 0:
        ranked = ranked[:top_k]
    return ranked


def prepare_training_frame(
    raw_df: pd.DataFrame, target_col: str, top_feature_count: int | None
) -> tuple[pd.DataFrame, list[str]]:
    """Build final training frame with selected features and target as last column."""
    engineered = apply_feature_engineering(raw_df)
    feature_candidates = build_feature_candidates(engineered)
    selected_features = select_top_correlated_features(
        engineered, target_col=target_col, candidates=feature_candidates, top_k=top_feature_count
    )

    training_frame = engineered[selected_features + [target_col]].dropna().reset_index(drop=True)
    return training_frame, selected_features


def create_data_splits(
    frame: pd.DataFrame,
    dataset_cls,
    seq_len: int = 24,
    test_size: float = 0.2,
    augment_factor: int = 1,
):
    """Create train/test split after scaling and sequence conversion."""
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    scaled_values = scaler.fit_transform(frame.values)

    base_dataset_eval = dataset_cls(scaled_values, seq_len=seq_len, augment=False)
    base_dataset_train = dataset_cls(scaled_values, seq_len=seq_len, augment=(augment_factor > 1))

    total_samples = len(base_dataset_eval)
    if total_samples < 10:
        raise ValueError(f"Not enough sequence samples after preprocessing: {total_samples}")

    test_samples = max(1, int(total_samples * test_size))
    train_samples = total_samples - test_samples
    if train_samples < 2:
        raise ValueError(f"Not enough training samples after split: {train_samples}")

    generator = torch.Generator()
    generator.manual_seed(42)
    all_indices = torch.randperm(total_samples, generator=generator)
    test_indices = all_indices[:test_samples].tolist()
    train_indices = all_indices[test_samples:].tolist()

    train_dataset = torch.utils.data.Subset(base_dataset_train, train_indices)
    test_dataset = torch.utils.data.Subset(base_dataset_eval, test_indices)

    return train_dataset, test_dataset, scaler, frame.shape[1] - 1


def build_dataloaders(train_dataset, test_dataset, batch_size: int):
    """Build train/val/test dataloaders with the same strategy as existing scripts."""
    if len(train_dataset) < 2:
        raise ValueError("Train dataset is too small to create a validation split.")

    train_size = max(1, int(len(train_dataset) * 0.85))
    if train_size >= len(train_dataset):
        train_size = len(train_dataset) - 1
    val_size = len(train_dataset) - train_size
    if val_size <= 0:
        raise ValueError("Validation dataset size became zero.")

    loader_train_indices = list(range(train_size))
    loader_val_indices = list(range(train_size, len(train_dataset)))

    train_subset = torch.utils.data.Subset(train_dataset, loader_train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, loader_val_indices)

    is_windows = platform.system() == "Windows"
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
        num_workers = max((os.cpu_count() or 2) - 1, 1)
        pin_memory = False
        persistent_workers = False
        prefetch_factor = 2

    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": True,
        "generator": generator,
    }
    if not is_windows:
        train_loader_kwargs["persistent_workers"] = persistent_workers
        train_loader_kwargs["prefetch_factor"] = prefetch_factor

    eval_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if not is_windows:
        eval_loader_kwargs["persistent_workers"] = persistent_workers
        eval_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_subset, **train_loader_kwargs)
    val_loader = DataLoader(val_subset, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, **eval_loader_kwargs)
    return train_loader, val_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_save_path: Path,
    epochs: int,
    lr: float,
    patience: int,
):
    """Train with the same optimizer/scheduler/loss strategy as current Auto-Mamformer scripts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    use_cuda = device.type == "cuda"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=5e-4,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=lr * 0.01
    )

    mse_criterion = nn.MSELoss()
    huber_criterion = nn.SmoothL1Loss(beta=0.5)
    scaler = GradScaler(enabled=use_cuda)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(epochs):
        if use_cuda:
            torch.cuda.synchronize()
        start_time = time.time()

        model.train()
        running_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad()

            amp_context = autocast(dtype=torch.float16) if use_cuda else nullcontext()
            with amp_context:
                pred = model(batch_x)
                mse_loss = mse_criterion(pred.squeeze(), batch_y.squeeze())
                huber_loss = huber_criterion(pred.squeeze(), batch_y.squeeze())
                loss = 0.7 * mse_loss + 0.3 * huber_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()

        model.eval()
        running_val_loss = 0.0
        val_preds = []
        val_trues = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                amp_context = autocast(dtype=torch.float16) if use_cuda else nullcontext()
                with amp_context:
                    pred = model(batch_x)
                    val_loss = mse_criterion(pred.squeeze(), batch_y.squeeze())

                running_val_loss += val_loss.item()
                val_preds.append(pred.detach().cpu())
                val_trues.append(batch_y.detach().cpu())

        train_loss = running_train_loss / max(1, len(train_loader))
        val_loss = running_val_loss / max(1, len(val_loader))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        val_pred_arr = torch.cat(val_preds).squeeze().numpy()
        val_true_arr = torch.cat(val_trues).squeeze().numpy()
        if np.size(val_true_arr) > 1:
            val_r2 = r2_score(val_true_arr, val_pred_arr)
        else:
            val_r2 = float("nan")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1

        if use_cuda:
            torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        print(
            f"Epoch [{epoch + 1:3d}/{epochs}] "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"R2: {val_r2:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    model.load_state_dict(torch.load(model_save_path, map_location=device))
    return model, train_losses, val_losses


def get_rescaled_predictions(model: nn.Module, data_loader: DataLoader, scaler: RobustScaler):
    """Run prediction and inverse-transform target."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    predictions = []
    true_values = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            output = model(batch_x)
            predictions.extend(output.cpu().numpy().flatten())
            true_values.extend(batch_y.cpu().numpy().flatten())

    predictions = np.array(predictions)
    true_values = np.array(true_values)

    dummy_pred = np.zeros((len(predictions), scaler.n_features_in_))
    dummy_true = np.zeros((len(true_values), scaler.n_features_in_))
    dummy_pred[:, -1] = predictions
    dummy_true[:, -1] = true_values

    pred_rescaled = scaler.inverse_transform(dummy_pred)[:, -1]
    true_rescaled = scaler.inverse_transform(dummy_true)[:, -1]
    return pred_rescaled, true_rescaled


def evaluate_and_save(
    predictions: np.ndarray,
    true_values: np.ndarray,
    target_name: str,
    plot_path: Path,
):
    """Compute metrics and save figure."""
    r2 = r2_score(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mask = np.abs(true_values) > 1e-8
    mape = float(np.mean(np.abs((true_values[mask] - predictions[mask]) / true_values[mask])) * 100)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(true_values[:200], label=f"True {target_name}", alpha=0.7)
    plt.plot(predictions[:200], label=f"Predicted {target_name}", alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel(target_name)
    plt.title(f"Auto-Mamformer BSM2: {target_name} Prediction (first 200)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(true_values, predictions, alpha=0.5)
    lim_min = float(min(np.min(true_values), np.min(predictions)))
    lim_max = float(max(np.max(true_values), np.max(predictions)))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=2)
    plt.xlabel(f"True {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"Prediction vs True (R2 = {r2:.4f})")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "r2": float(r2),
        "mape": float(mape),
        "mae": float(mae),
        "rmse": float(rmse),
    }


def run_single_target(
    target_key: str,
    args: argparse.Namespace,
    AutoMamformerModel,
    TimeSeriesDataset,
    set_seed_fn,
):
    """Train/evaluate one target (`cod` or `bod`)."""
    target_map = {
        "cod": ("eff_COD", "COD"),
        "bod": ("eff_BOD5", "BOD5"),
    }
    if target_key not in target_map:
        raise ValueError(f"Unsupported target: {target_key}")

    target_col, target_label = target_map[target_key]
    print("\n" + "=" * 70)
    print(f"Auto-Mamformer BSM2 - {target_label} prediction")
    print("=" * 70)

    set_seed_fn(args.seed)
    raw_df = pd.read_csv(args.data_path)
    raw_df = add_bsm2_targets(raw_df)

    if target_col not in raw_df.columns:
        raise ValueError(f"Target column not found after target construction: {target_col}")

    frame, selected_features = prepare_training_frame(
        raw_df, target_col=target_col, top_feature_count=args.top_features
    )
    if args.max_samples is not None and args.max_samples > 0 and len(frame) > args.max_samples:
        frame = frame.iloc[: args.max_samples].copy().reset_index(drop=True)

    print(f"Data rows: {len(frame)} | Input features: {len(selected_features)} | Target: {target_col}")

    train_dataset, test_dataset, scaler, input_dim = create_data_splits(
        frame,
        dataset_cls=TimeSeriesDataset,
        seq_len=args.seq_len,
        test_size=args.test_size,
        augment_factor=args.augment_factor,
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        train_dataset, test_dataset, batch_size=args.batch_size
    )

    model = AutoMamformerModel(
        input_dim=input_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        seq_len=args.seq_len,
        pred_len=1,
        dropout=args.dropout,
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    model_save_path = Path(args.model_dir) / f"auto_mamformer_bsm2_{target_key}.pth"
    result_plot_path = Path(args.result_dir) / f"auto_mamformer_bsm2_{target_key}_results.png"
    result_npy_path = Path(args.result_dir) / f"auto_mamformer_bsm2_{target_key}_results.npy"

    model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_save_path=model_save_path,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )

    pred_rescaled, true_rescaled = get_rescaled_predictions(model, test_loader, scaler)
    metrics = evaluate_and_save(
        predictions=pred_rescaled,
        true_values=true_rescaled,
        target_name=target_label,
        plot_path=result_plot_path,
    )

    results = {
        **metrics,
        "target": target_col,
        "selected_features": selected_features,
        "predictions": pred_rescaled,
        "true_values": true_rescaled,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    np.save(result_npy_path, results)

    print(f"Saved model: {model_save_path}")
    print(f"Saved plot:  {result_plot_path}")
    print(f"Saved npy:   {result_npy_path}")
    print(
        f"Metrics => R2: {metrics['r2']:.4f} | "
        f"MAPE: {metrics['mape']:.2f}% | MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f}"
    )
    return metrics


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Auto-Mamformer BSM2 COD/BOD5 training entry.")
    parser.add_argument("--data-path", default="data/BSM2/bsm2_full_data.csv")
    parser.add_argument("--target", choices=("cod", "bod", "both"), default="both")

    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--augment-factor", type=int, default=2)
    parser.add_argument("--top-features", type=int, default=60)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--result-dir", default="result")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    AutoMamformerModel, TimeSeriesDataset, set_seed_fn = load_auto_mamformer_components()

    targets = ["cod", "bod"] if args.target == "both" else [args.target]
    summary = {}
    for target_key in targets:
        metrics = run_single_target(
            target_key=target_key,
            args=args,
            AutoMamformerModel=AutoMamformerModel,
            TimeSeriesDataset=TimeSeriesDataset,
            set_seed_fn=set_seed_fn,
        )
        summary[target_key] = metrics

    summary_path = Path(args.result_dir) / "auto_mamformer_bsm2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
