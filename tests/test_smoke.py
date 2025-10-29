"""
Smoke test: instantiate encoder+unet and run a forward and a tiny sample loop.
"""

import torch

from aud2studio.config import RootCfg
from aud2studio.cond import AcousticEncoder
from aud2studio.model import build_unet


def test_shapes():
    cfg = RootCfg()
    unet = build_unet(cfg.model, cfg.mel)
    enc = AcousticEncoder(dim=cfg.model.conditioning_dim)

    b, m, n = 2, cfg.mel.n_mels, 64
    x = torch.randn(b, 1, m, n)
    t = torch.randint(0, 1000, (b,))
    cond = enc(x)
    y = unet(x, t, encoder_hidden_states=cond).sample
    assert y.shape == x.shape
