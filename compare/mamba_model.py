import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMambaBlock(nn.Module):
    """简化版Mamba块"""

    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.input_proj = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=3,
            padding=1,
            groups=d_model,
        )
        self.gate = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        u, v = self.input_proj(x).chunk(2, dim=-1)
        v = torch.tanh(v)

        conv_in = u.transpose(1, 2)  # [B, C, L]
        conv_out = self.conv(conv_in).transpose(1, 2)  # [B, L, C]
        conv_out = F.silu(conv_out)

        gated = conv_out * torch.sigmoid(self.gate(v))
        out = self.output_proj(gated)
        out = self.dropout(out)
        return residual + out


class MambaRegressor(nn.Module):
    """简化Mamba序列回归"""

    def __init__(self, input_dim: int, d_model: int = 128, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(512, d_model) * 0.01)
        self.layers = nn.ModuleList([
            SimpleMambaBlock(d_model=d_model, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        b, l, _ = x.shape
        x = self.input_proj(x)
        if l <= self.pos_embedding.size(0):
            x = x + self.pos_embedding[:l]
        else:
            pos = F.interpolate(self.pos_embedding[:1].transpose(0, 1).unsqueeze(0), size=l, mode="linear", align_corners=False)
            pos = pos.squeeze(0).transpose(0, 1)
            x = x + pos
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        last = x[:, -1, :]
        pred = self.head(last)
        return pred


