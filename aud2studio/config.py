"""
@author aud2studio maintainers <maintainers@example.com>
@description Pydantic config dataclasses and YAML loader.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class AudioCfg(BaseModel):
    sample_rate: int = 48000
    clip_seconds: float = 10.0
    mono: bool = True


class MelCfg(BaseModel):
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    fmin: int = 20
    fmax: int = 20000


class ModelCfg(BaseModel):
    base_channels: int = 128
    layers: int = 4
    attention_resolutions: List[int] = Field(default_factory=lambda: [16, 8, 4])
    conditioning_dim: int = 256


class SchedulerCfg(BaseModel):
    name: str = "ddpm"
    num_inference_steps: int = 50


class TrainCfg(BaseModel):
    batch_size: int = 8
    lr: float = 2.0e-4
    ema: bool = True
    grad_clip: float = 1.0
    epochs: int = 100
    optimizer: str = "adamw"
    fp16: bool = True
    output_dir: str = "./runs"
    save_every_steps: int = 1000
    save_total_limit: int = 5


class DataCfg(BaseModel):
    rir_dir: str = "./data/rirs"
    noise_dir: str = "./data/noise"
    clean_music_dir: str = "./data/clean"
    pairs_manifest: str = "./data/pairs.jsonl"


class EvalCfg(BaseModel):
    metrics: List[str] = Field(default_factory=lambda: ["si_sdr", "srmr", "fad"])
    val_every_steps: int = 1000


class ExportCfg(BaseModel):
    half_precision: bool = True


class RootCfg(BaseModel):
    experiment: str = "bootleg_v1"
    seed: int = 1337
    audio: AudioCfg = AudioCfg()
    mel: MelCfg = MelCfg()
    model: ModelCfg = ModelCfg()
    scheduler: SchedulerCfg = SchedulerCfg()
    train: TrainCfg = TrainCfg()
    data: DataCfg = DataCfg()
    eval: EvalCfg = EvalCfg()
    export: ExportCfg = ExportCfg()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RootCfg":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def to_dict(self) -> dict:
        return self.model_dump()
