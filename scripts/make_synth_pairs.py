"""
@author aud2studio maintainers <maintainers@example.com>
@description Make synthetic audience/clean pairs by convolving clean clips with random RIRs and adding noise.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torchaudio


def list_wavs(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.wav")])


def random_crop(wav: torch.Tensor, target: int) -> torch.Tensor:
    if wav.numel() <= target:
        return torch.nn.functional.pad(wav, (0, target - wav.numel()))
    start = random.randint(0, wav.numel() - target)
    return wav[start : start + target]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clean_dir", type=str, required=True)
    p.add_argument("--rir_dir", type=str, required=True)
    p.add_argument("--noise_dir", type=str, required=True)
    p.add_argument("--out_manifest", type=str, required=True)
    p.add_argument("--sample_rate", type=int, default=48000)
    p.add_argument("--clip_seconds", type=float, default=10.0)
    p.add_argument("--num", type=int, default=100)
    args = p.parse_args()

    sr = args.sample_rate
    target = int(sr * args.clip_seconds)

    cleans = list_wavs(Path(args.clean_dir))
    rirs = list_wavs(Path(args.rir_dir))
    noises = list_wavs(Path(args.noise_dir))

    out = Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for i in range(args.num):
            clean_p = random.choice(cleans)
            rir_p = random.choice(rirs)
            noise_p = random.choice(noises)

            clean, sr_c = torchaudio.load(str(clean_p))
            if sr_c != sr:
                clean = torchaudio.functional.resample(clean, sr_c, sr)
            clean = torch.mean(clean, dim=0)
            clean = random_crop(clean, target)

            rir, sr_r = torchaudio.load(str(rir_p))
            if sr_r != sr:
                rir = torchaudio.functional.resample(rir, sr_r, sr)
            rir = torch.mean(rir, dim=0)

            aud = torch.nn.functional.conv1d(
                clean.view(1, 1, -1), rir.view(1, 1, -1), padding=rir.numel() - 1
            ).view(-1)
            aud = random_crop(aud, target)

            noise, sr_n = torchaudio.load(str(noise_p))
            if sr_n != sr:
                noise = torchaudio.functional.resample(noise, sr_n, sr)
            noise = torch.mean(noise, dim=0)
            noise = random_crop(noise, target)
            snr_db = random.uniform(0.0, 20.0)
            noise = noise * (10 ** (-snr_db / 20.0))
            aud = aud + noise

            # write temp wavs
            aud_out = out.parent / f"aud_{i:06d}.wav"
            clean_out = out.parent / f"clean_{i:06d}.wav"
            torchaudio.save(str(aud_out), aud.unsqueeze(0), sr)
            torchaudio.save(str(clean_out), clean.unsqueeze(0), sr)

            f.write(json.dumps({"aud": str(aud_out), "ref": str(clean_out)}) + "\n")

    print(f"Wrote manifest to {out}")


if __name__ == "__main__":
    main()
