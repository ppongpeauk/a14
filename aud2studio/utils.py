"""
@author aud2studio maintainers <maintainers@example.com>
@description Utility helpers for seeding, audio I/O, and tensor ops.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchaudio


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_audio(path: str | Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)  # [T]


def save_audio(path: str | Path, wav: torch.Tensor, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().cpu().unsqueeze(0)
    torchaudio.save(str(path), wav, sr)


def rms_normalize(
    wav: torch.Tensor, target_db: float = -23.0, eps: float = 1e-8
) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(wav**2) + eps)
    target = 10 ** (target_db / 20.0)
    return wav * (target / (rms + eps))


def pad_or_trim(wav: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    if wav.numel() >= target_num_samples:
        return wav[:target_num_samples]
    pad = target_num_samples - wav.numel()
    return torch.nn.functional.pad(wav, (0, pad))


def shape_str(x: torch.Tensor) -> str:
    return "x[" + ",".join(str(s) for s in x.shape) + "]"
