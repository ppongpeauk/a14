"""
@author aud2studio maintainers <maintainers@example.com>
@description Train/eval loops with accelerate. DDPM noise prediction objective.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil

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


def _prune_checkpoints(root: Path, keep: int | None) -> None:
    if keep is None or keep <= 0:
        return
    ckpts = sorted(
        [p for p in root.glob("step-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    for old in ckpts[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


def save_checkpoint(
    accel: Accelerator,
    *,
    step: int,
    epoch: int,
    unet,
    cond_enc,
    opt,
    cfg: RootCfg,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / f"step-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "epoch": epoch,
        "config": cfg.to_dict(),
        "unet": accel.unwrap_model(unet).state_dict(),
        "cond": accel.unwrap_model(cond_enc).state_dict(),
        "optimizer": opt.state_dict(),
    }
    accel.save(state, ckpt_dir / "checkpoint.pt")
    # prune
    _prune_checkpoints(out_dir, cfg.train.save_total_limit)


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

    # checkpoint root: <output_dir>/<experiment>
    ckpt_root = Path(cfg.train.output_dir) / cfg.experiment

    for epoch in range(cfg.train.epochs):
        pbar = tqdm(
            dl,
            disable=not accel.is_local_main_process,
            desc=f"epoch {epoch + 1}/{cfg.train.epochs}",
        )
        for batch in pbar:
            wav_in = batch["aud"].to(accel.device)
            wav_tgt = batch["ref"].to(accel.device)

            mel_in = wav_to_mel(wav_in, cfg.mel, cfg.audio.sample_rate)  # [B, 1, M, N]
            mel_tgt = wav_to_mel(wav_tgt, cfg.mel, cfg.audio.sample_rate)

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
            # periodic checkpoint
            if (
                accel.is_local_main_process
                and cfg.train.save_every_steps > 0
                and ts.step % cfg.train.save_every_steps == 0
            ):
                save_checkpoint(
                    accel,
                    step=ts.step,
                    epoch=epoch,
                    unet=unet,
                    cond_enc=cond_enc,
                    opt=opt,
                    cfg=cfg,
                    out_dir=ckpt_root,
                )

    # final checkpoint
    if accel.is_local_main_process:
        save_checkpoint(
            accel,
            step=ts.step,
            epoch=cfg.train.epochs - 1,
            unet=unet,
            cond_enc=cond_enc,
            opt=opt,
            cfg=cfg,
            out_dir=ckpt_root,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    cfg = RootCfg.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
