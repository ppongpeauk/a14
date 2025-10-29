"""
@author aud2studio maintainers <maintainers@example.com>
@description Metrics: SI-SDR (native), SRMR & FAD stubs with optional deps.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def si_sdr(x: torch.Tensor, s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant SDR for single-channel waveforms.
    x: estimate [B, T], s: reference [B, T]
    returns: [B] SI-SDR in dB
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if s.dim() == 1:
        s = s.unsqueeze(0)
    s_target = (
        torch.sum(x * s, dim=1, keepdim=True)
        * s
        / (torch.sum(s**2, dim=1, keepdim=True) + eps)
    )
    e_noise = x - s_target
    ratio = (torch.sum(s_target**2, dim=1) + eps) / (torch.sum(e_noise**2, dim=1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def srmr(wav: torch.Tensor, sr: int) -> Optional[torch.Tensor]:
    try:
        from srmrpy import srmr as srmr_fn
    except Exception:
        return None
    vals = []
    for i in range(wav.size(0)):
        v, _ = srmr_fn(wav[i].detach().cpu().numpy(), sr)
        vals.append(torch.tensor(v, dtype=torch.float32))
    return torch.stack(vals, dim=0)


def fad(
    embeddings_ref: torch.Tensor, embeddings_est: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    Placeholder: requires external embedding model and FAD calculation.
    Return None to indicate not available without extras.
    """
    return None
