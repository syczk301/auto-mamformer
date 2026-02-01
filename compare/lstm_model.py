import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """简单的LSTM回归模型"""

    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # 使用最后一个时间步
        pred = self.head(last)
        return pred


