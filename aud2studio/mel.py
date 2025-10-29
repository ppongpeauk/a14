"""
@author aud2studio maintainers <maintainers@example.com>
@description STFT/mel helpers and Griffin-Lim vocoder fallback.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torchaudio

from .config import MelCfg


def wav_to_mel(wav: torch.Tensor, cfg: MelCfg, sr: int = 48000) -> torch.Tensor:
    """
    Convert mono waveform tensor [B, T] or [T] to log-mel [B, 1, M, N].
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,  # caller must ensure resample beforehand
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
    n_freqs = cfg.n_fft // 2 + 1
    f_max = min(cfg.fmax, sr / 2.0)
    try:
        fbanks = torchaudio.functional.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=cfg.fmin,
            f_max=f_max,
            n_mels=cfg.n_mels,
            sample_rate=sr,
            norm=None,
            mel_scale="htk",
        )  # [F, M]
    except AttributeError:
        fbanks = torchaudio.functional.create_fb_matrix(
            n_freqs=n_freqs,
            f_min=cfg.fmin,
            f_max=f_max,
            n_mels=cfg.n_mels,
            sample_rate=sr,
            norm=None,
            mel_scale="htk",
        )  # [F, M]
    mel_fb = fbanks.to(mel_spec.device, mel_spec.dtype).T  # [M, F]
    pinv = torch.linalg.pinv(mel_fb)  # [F, M]
    return pinv @ mel_spec  # [F, N]


def mel_to_wav_vocoder(
    log_mel: torch.Tensor, cfg: MelCfg, sr: int = 48000, n_iter: int | None = None
) -> torch.Tensor:
    """
    Invert log-mel [B, 1, M, N] to waveform [B, T] using Griffin-Lim as a fallback vocoder.
    """
    b = log_mel.size(0)
    # our mel features are power (because power=2.0 in wav_to_mel)
    mel_power = torch.exp(log_mel.squeeze(1))  # [B, M, N]
    # Prefer InverseMelScale if available
    mags_power_list = []
    use_inverse = cfg.invert_method in ("auto", "inverse_mel_scale")
    inv_ok = False
    if use_inverse:
        try:
            inv = torchaudio.transforms.InverseMelScale(
                n_stft=cfg.n_fft // 2 + 1,
                n_mels=cfg.n_mels,
                sample_rate=sr,
                f_min=cfg.fmin,
                f_max=min(cfg.fmax, sr / 2.0),
                max_iter=0,
                norm=None,
                mel_scale="htk",
            ).to(device=mel_power.device, dtype=mel_power.dtype)
            for i in range(b):
                mags_power_list.append(inv(mel_power[i]))
            inv_ok = True
        except Exception:
            inv_ok = False
    if not inv_ok:
        # fallback: pseudo-inverse of mel basis to linear power
        for i in range(b):
            mags_power_list.append(_mel_to_mag(mel_power[i], cfg, sr))
    mags_power = torch.stack(mags_power_list, dim=0)  # [B, F, N]
    mags = torch.sqrt(
        torch.clamp(mags_power, min=1e-8)
    )  # convert to magnitude for Griffin-Lim
    # Griffin-Lim on linear magnitude
    iters = int(cfg.gl_iters if n_iter is None else n_iter)
    griffin = torchaudio.transforms.GriffinLim(
        n_fft=cfg.n_fft, hop_length=cfg.hop_length, n_iter=iters
    ).to(device=mags.device, dtype=mags.dtype)
    wavs = []
    for i in range(b):
        wav = griffin(mags[i])
        wavs.append(wav)
    wav = torch.stack(wavs, dim=0)
    return wav
