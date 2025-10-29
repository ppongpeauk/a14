"""
@author aud2studio maintainers <maintainers@example.com>
@description Train/eval loops with accelerate. DDPM noise prediction objective.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from accelerate import Accelerator
from diffusers import DDPMScheduler
from tqdm.auto import tqdm

from .config import RootCfg
from .data import PairDataset, collate_audio
from .mel import wav_to_mel
from .cond import AcousticEncoder
from .model import build_unet
from .utils import set_seed


@dataclass
class TrainState:
    step: int = 0


def train(cfg: RootCfg) -> None:
    set_seed(cfg.seed)
    accel = Accelerator(mixed_precision="fp16" if cfg.train.fp16 else "no")

    unet = build_unet(cfg.model, cfg.mel)
    cond_enc = AcousticEncoder(dim=cfg.model.conditioning_dim)
    noise_sched = DDPMScheduler(num_train_timesteps=1000)
    ds = PairDataset(cfg.data, cfg.audio)

    opt = torch.optim.AdamW(unet.parameters(), lr=cfg.train.lr)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_audio,
    )

    unet, cond_enc, opt, dl = accel.prepare(unet, cond_enc, opt, dl)

    unet.train()
    cond_enc.train()
    ts = TrainState()

    for epoch in range(cfg.train.epochs):
        pbar = tqdm(
            dl,
            disable=not accel.is_local_main_process,
            desc=f"epoch {epoch + 1}/{cfg.train.epochs}",
        )
        for batch in pbar:
            wav_in = batch["aud"].to(accel.device)
            wav_tgt = batch["ref"].to(accel.device)

            mel_in = wav_to_mel(wav_in, cfg.mel)  # [B, 1, M, N]
            mel_tgt = wav_to_mel(wav_tgt, cfg.mel)

            cond = cond_enc(mel_in)  # [B, 1, C]

            noise = torch.randn_like(mel_tgt)
            t = torch.randint(
                0,
                noise_sched.config.num_train_timesteps,
                (mel_tgt.size(0),),
                device=mel_tgt.device,
            )
            noisy = noise_sched.add_noise(mel_tgt, noise, t)
            pred = unet(noisy, t, encoder_hidden_states=cond).sample
            loss = torch.mean((pred - noise) ** 2)

            accel.backward(loss)
            if cfg.train.grad_clip is not None and cfg.train.grad_clip > 0:
                accel.clip_grad_norm_(unet.parameters(), cfg.train.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

            ts.step += 1
            if accel.is_local_main_process:
                pbar.set_postfix(loss=f"{loss.item():.4f}", step=ts.step)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    cfg = RootCfg.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
