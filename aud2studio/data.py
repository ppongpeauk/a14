"""
@author aud2studio maintainers <maintainers@example.com>
@description Dataset for paired audience/clean audio with simple augment and RIR loader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio

from .config import DataCfg, AudioCfg
from .utils import load_audio, pad_or_trim, rms_normalize


class PairDataset(torch.utils.data.Dataset):
    """
    Reads JSONL manifest with lines of the form:
      {"aud": ["/path/a.wav", "/path/b.wav"], "ref": "/path/clean.wav"}
    Multiple audience paths are optional; if multiple present, they are mixed (mean).
    """

    def __init__(self, data_cfg: DataCfg, audio_cfg: AudioCfg) -> None:
        super().__init__()
        self.manifest = Path(data_cfg.pairs_manifest)
        self.audio_cfg = audio_cfg
        self._items: List[Dict] = []
        with open(self.manifest, "r") as f:
            for line in f:
                if line.strip():
                    self._items.append(json.loads(line))

        self.target_num_samples = int(audio_cfg.sample_rate * audio_cfg.clip_seconds)

    def __len__(self) -> int:
        return len(self._items)

    def _mix_audience(self, paths: List[str]) -> torch.Tensor:
        waves = []
        for p in paths:
            w = load_audio(p, self.audio_cfg.sample_rate)
            waves.append(w)
        if not waves:
            raise RuntimeError("No audience paths in manifest entry")
        wav = torch.stack(waves, dim=0).mean(dim=0)
        return wav

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self._items[idx]
        aud_paths = item["aud"] if isinstance(item["aud"], list) else [item["aud"]]
        ref_path = item["ref"]

        wav_a = self._mix_audience(aud_paths)
        wav_r = load_audio(ref_path, self.audio_cfg.sample_rate)

        wav_a = rms_normalize(wav_a)
        wav_r = rms_normalize(wav_r)

        wav_a = pad_or_trim(wav_a, self.target_num_samples)
        wav_r = pad_or_trim(wav_r, self.target_num_samples)

        return {"aud": wav_a, "ref": wav_r}


def collate_audio(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    aud = torch.stack([b["aud"] for b in batch], dim=0)
    ref = torch.stack([b["ref"] for b in batch], dim=0)
    return {"aud": aud, "ref": ref}
