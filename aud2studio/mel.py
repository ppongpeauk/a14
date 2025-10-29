"""
@author aud2studio maintainers <maintainers@example.com>
@description STFT/mel helpers and Griffin-Lim vocoder fallback.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torchaudio

from .config import MelCfg


def wav_to_mel(wav: torch.Tensor, cfg: MelCfg) -> torch.Tensor:
    """
    Convert mono waveform tensor [B, T] or [T] to log-mel [B, 1, M, N].
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=48000,  # caller must ensure resample beforehand
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.fmin,
        f_max=cfg.fmax,
        center=True,
        power=2.0,
        norm=None,
        mel_scale="htk",
    ).to(wav.device)(
        wav
    )  # [B, M, N]
    logmel = torch.log(spec + 1e-6)
    return logmel.unsqueeze(1)


def _mel_to_mag(mel_spec: torch.Tensor, cfg: MelCfg, sr: int = 48000) -> torch.Tensor:
    """
    Approximate inverse mel filter to linear magnitude spectrogram.
    mel_spec: [B, M, N] (linear magnitude, not log)
    returns: [B, F, N]
    """
    mel_fb = torchaudio.functional.create_fb_matrix(
        n_freqs=cfg.n_fft // 2 + 1,
        f_min=cfg.fmin,
        f_max=cfg.fmax,
        n_mels=cfg.n_mels,
        sample_rate=sr,
        norm=None,
        mel_scale="htk",
    ).to(
        mel_spec.device, mel_spec.dtype
    )  # [F, M]
    # Pseudo-inverse
    pinv = torch.linalg.pinv(mel_fb)
    return torch.matmul(pinv, mel_spec)  # [F, N]


def mel_to_wav_vocoder(
    log_mel: torch.Tensor, cfg: MelCfg, sr: int = 48000, n_iter: int = 32
) -> torch.Tensor:
    """
    Invert log-mel [B, 1, M, N] to waveform [B, T] using Griffin-Lim as a fallback vocoder.
    """
    b = log_mel.size(0)
    mel = torch.exp(log_mel.squeeze(1))  # [B, M, N]
    mags = torch.stack([_mel_to_mag(mel[i], cfg, sr) for i in range(b)], dim=0)
    # Griffin-Lim on linear magnitude
    griffin = torchaudio.transforms.GriffinLim(
        n_fft=cfg.n_fft, hop_length=cfg.hop_length, n_iter=n_iter
    )
    wavs = []
    for i in range(b):
        wav = griffin(mags[i])
        wavs.append(wav)
    wav = torch.stack(wavs, dim=0)
    return wav
