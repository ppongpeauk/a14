"""
@author aud2studio maintainers <maintainers@example.com>
@description CLI inference: load audio(s), compute mel/condition, sample with scheduler, invert to wav.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from diffusers import DDPMScheduler

from .config import RootCfg
from .mel import wav_to_mel, mel_to_wav_vocoder
from .cond import AcousticEncoder
from .model import build_unet
from .utils import load_audio, rms_normalize, save_audio, pad_or_trim


@torch.no_grad()
def sample_with_scheduler(
    unet, cond: torch.Tensor, cfg: RootCfg, shape: torch.Size
) -> torch.Tensor:
    """
    Basic DDPM sampling: start from noise and denoise with conditioning.
    Returns: log-mel [B, 1, M, N]
    """
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(cfg.scheduler.num_inference_steps)
    x = torch.randn(shape, device=cond.device)
    for t in scheduler.timesteps:
        noise_pred = unet(x, t, encoder_hidden_states=cond).sample
        x = scheduler.step(noise_pred, t, x).prev_sample
    return x


def run(inputs: List[str], out_path: str, cfg: RootCfg) -> None:
    sr = cfg.audio.sample_rate
    target_len = int(cfg.audio.clip_seconds * sr)
    waves = [pad_or_trim(rms_normalize(load_audio(p, sr)), target_len) for p in inputs]
    mix = torch.stack(waves, dim=0).mean(dim=0).unsqueeze(0)  # [1, T]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = build_unet(cfg.model, cfg.mel).to(device).eval()
    cond_enc = AcousticEncoder(dim=cfg.model.conditioning_dim).to(device).eval()

    mel_in = wav_to_mel(mix.to(device), cfg.mel)
    cond = cond_enc(mel_in)  # [1, 1, C]

    # sample log-mel
    b, _, m, n = mel_in.shape
    log_mel = sample_with_scheduler(unet, cond, cfg, shape=mel_in.shape)
    wav = mel_to_wav_vocoder(log_mel, cfg.mel, sr)

    save_audio(out_path, wav.squeeze(0).cpu(), sr)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", type=str, required=True)
    p.add_argument("--save", type=str, required=True)
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()

    cfg = RootCfg.from_yaml(args.config)
    run(args.inputs, args.save, cfg)


if __name__ == "__main__":
    main()
