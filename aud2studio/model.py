"""
@author aud2studio maintainers <maintainers@example.com>
@description UNet2DConditionModel wiring for mel "images" with cross-attention.
"""

from __future__ import annotations

from typing import List

from diffusers import UNet2DConditionModel

from .config import ModelCfg, MelCfg


def build_unet(cfg: ModelCfg, mel: MelCfg) -> UNet2DConditionModel:
    """
    Configure a Diffusers UNet2DConditionModel for mel inputs.
    sample_size is set to the mel bin count; width (time) can be variable.
    """
    chs: List[int] = [cfg.base_channels * (2**i) for i in range(cfg.layers)]
    unet = UNet2DConditionModel(
        sample_size=mel.n_mels,
        in_channels=1,
        out_channels=1,
        down_block_types=(
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            *("DownBlock2D",) * max(0, cfg.layers - 2),
        ),
        up_block_types=(
            *("UpBlock2D",) * max(0, cfg.layers - 2),
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels=chs,
        cross_attention_dim=cfg.conditioning_dim,
        layers_per_block=2,
    )
    return unet
