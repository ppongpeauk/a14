"""
@author aud2studio maintainers <maintainers@example.com>
@description Dataset for paired audience/clean audio with simple augment and RIR loader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        # Segmentation config
        self.segment_seconds: Optional[float] = getattr(data_cfg, "segment_seconds", None)
        self.hop_seconds: Optional[float] = getattr(data_cfg, "hop_seconds", None)
        self.drop_last_short: bool = getattr(data_cfg, "drop_last_short", False)

        self._segments: List[Tuple[int, int]] | None = None
        if self.segment_seconds is not None:
            self._segments = self._build_segments()

    def __len__(self) -> int:
        if self._segments is not None:
            return len(self._segments)
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
        if self._segments is not None:
            item_idx, start = self._segments[idx]
            item = self._items[item_idx]
            seg_len = int(self.segment_seconds * self.audio_cfg.sample_rate)
        else:
            item = self._items[idx]
            start = 0
            seg_len = self.target_num_samples
        aud_paths = item["aud"] if isinstance(item["aud"], list) else [item["aud"]]
        ref_path = item["ref"]

        wav_a = self._mix_audience(aud_paths)
        wav_r = load_audio(ref_path, self.audio_cfg.sample_rate)

        # segment window
        end = start + seg_len
        wav_a = wav_a[start:end]
        wav_r = wav_r[start:end]

        wav_a = pad_or_trim(wav_a, seg_len)
        wav_r = pad_or_trim(wav_r, seg_len)
        wav_a = rms_normalize(wav_a)
        wav_r = rms_normalize(wav_r)

        return {"aud": wav_a, "ref": wav_r}

    # --- helpers ---
    def _probe_len_samples(self, paths: List[str], ref_path: str) -> int:
        sr_tgt = self.audio_cfg.sample_rate
        lengths = []
        for p in paths + [ref_path]:
            try:
                info = torchaudio.info(p)
                frames = info.num_frames
                sr = info.sample_rate if info.sample_rate else sr_tgt
                est = int(round(frames * sr_tgt / sr))
                lengths.append(est)
            except Exception:
                # fallback: load header minimally
                wav = load_audio(p, sr_tgt)
                lengths.append(int(wav.numel()))
        return max(lengths) if lengths else 0

    def _build_segments(self) -> List[Tuple[int, int]]:
        seg_len = int(self.segment_seconds * self.audio_cfg.sample_rate)
        hop = int((self.hop_seconds or self.segment_seconds) * self.audio_cfg.sample_rate)
        segments: List[Tuple[int, int]] = []
        for i, item in enumerate(self._items):
            aud_paths = item["aud"] if isinstance(item["aud"], list) else [item["aud"]]
            ref_path = item["ref"]
            total = self._probe_len_samples(aud_paths, ref_path)
            if total <= 0:
                continue
            if total <= seg_len:
                segments.append((i, 0))
                if not self.drop_last_short:
                    # single padded segment; nothing else to add
                    pass
                continue
            # regular hops
            start = 0
            last_full_start = max(0, total - seg_len)
            while start <= last_full_start:
                segments.append((i, start))
                start += hop
            # tail
            if not self.drop_last_short and (len(segments) == 0 or segments[-1][1] != last_full_start):
                segments.append((i, last_full_start))
        return segments


def collate_audio(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    aud = torch.stack([b["aud"] for b in batch], dim=0)
    ref = torch.stack([b["ref"] for b in batch], dim=0)
    return {"aud": aud, "ref": ref}
