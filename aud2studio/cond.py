"""
@author aud2studio maintainers <maintainers@example.com>
@description Acoustic conditioning encoder: small Transformer over mel frames.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, max_len: int = 10000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class AcousticEncoder(nn.Module):
    """
    Encodes a sequence of mel frames into a fixed-dim conditioning vector.

    Inputs: log-mel [B, 1, M, N]
    Output: encoder_hidden_states [B, 1, D]
    """

    def __init__(self, dim: int = 256, num_layers: int = 2, num_heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Linear(
            128, dim
        )  # assumes 128 mel bins by default; will adapt via conv if different
        self.adapt_conv = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4, batch_first=True
        )
        self.pos = PositionalEncoding(dim)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        b, _, m, n = log_mel.shape
        x = log_mel.squeeze(1)  # [B, M, N]
        if m != 128:
            # lightweight adaptation to expected projection input
            x = torch.nn.functional.interpolate(
                x.unsqueeze(1), size=(128, n), mode="bilinear", align_corners=False
            ).squeeze(1)
            m = 128
        x = x.transpose(1, 2)  # [B, N, M]
        x = self.in_proj(x)  # [B, N, D]
        x = self.pos(x)
        x = self.enc(x)  # [B, N, D]
        # temporal mean pool
        x = x.mean(dim=1, keepdim=True)  # [B, 1, D]
        return x
