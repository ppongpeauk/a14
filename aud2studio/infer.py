"""
@author aud2studio maintainers <maintainers@example.com>
@description CLI inference: load audio(s), compute mel/condition, sample with scheduler, invert to wav.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import torch
from diffusers import DDPMScheduler
from tqdm.auto import tqdm

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
    for t in tqdm(scheduler.timesteps, desc="DDPM sampling"):
        noise_pred = unet(x, t, encoder_hidden_states=cond).sample
        x = scheduler.step(noise_pred, t, x).prev_sample
    return x


def run(
    inputs: List[str], out_path: str, cfg: RootCfg, checkpoint: Optional[str] = None
) -> None:
    sr = cfg.audio.sample_rate
    raw = [rms_normalize(load_audio(p, sr)) for p in inputs]
    max_len = max(w.numel() for w in raw)
    waves = [pad_or_trim(w, max_len) for w in raw]
    mix = torch.stack(waves, dim=0).mean(dim=0).unsqueeze(0)  # [1, T]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = build_unet(cfg.model, cfg.mel).to(device).eval()
    cond_enc = AcousticEncoder(dim=cfg.model.conditioning_dim).to(device).eval()

    # optional checkpoint loading
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        if ckpt_path.is_dir():
            cand = ckpt_path / "checkpoint.pt"
            if cand.exists():
                ckpt_path = cand
        try:
            state = torch.load(str(ckpt_path), map_location=device)
            if isinstance(state, dict) and "unet" in state:
                unet.load_state_dict(state["unet"])
                if "cond" in state:
                    cond_enc.load_state_dict(state["cond"])  # optional
            elif isinstance(state, dict):
                # assume it's the unet state_dict only
                unet.load_state_dict(state)
            else:
                print(f"[warn] Unrecognized checkpoint format at {ckpt_path}")
        except Exception as e:
            print(f"[warn] Failed to load checkpoint {ckpt_path}: {e}")

    mel_in = wav_to_mel(mix.to(device), cfg.mel, sr)
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
    p.add_argument(
        "--model", "--checkpoint", dest="checkpoint", type=str, required=False
    )
    args = p.parse_args()

    cfg = RootCfg.from_yaml(args.config)
    run(args.inputs, args.save, cfg, args.checkpoint)


if __name__ == "__main__":
    main()
