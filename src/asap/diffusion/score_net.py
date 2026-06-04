"""Small local point-cloud score network for ASAP VP-SDE purification.

This module intentionally keeps the first Track-A model compact:

- fixed-size local SPU patches, normalized by SPU center and radius;
- VP-SDE denoising-score objective on clean KITTI patches;
- PointNet-style shared MLP + global context;
- checkpoint helpers used by :class:`VPSDEPurifier`.

The model predicts the score `grad_x log p_t(x)` for a noised local patch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class ScoreNetConfig:
    patch_size: int = 64
    hidden_dim: int = 128
    time_dim: int = 32
    beta_min: float = 0.1
    beta_max: float = 20.0
    t_eps: float = 1e-3
    t_max: float = 0.20
    max_coord: float = 3.0


def alpha_bar(beta_min: float, beta_max: float, t):
    """VP-SDE survival coefficient for scalar or tensor `t`."""
    return (-beta_min * t - 0.5 * (beta_max - beta_min) * t * t).exp()


def np_alpha_bar(beta_min: float, beta_max: float, t: float) -> float:
    return float(np.exp(-beta_min * t - 0.5 * (beta_max - beta_min) * t * t))


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "Track A score-net support requires PyTorch. Use the `asap` conda env."
        ) from exc
    return torch, nn


class SinusoidalTimeEmbedding:
    """Callable sinusoidal embedding to avoid depending on a larger framework."""

    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, t):
        torch, _ = require_torch()
        half = self.dim // 2
        if half == 0:
            return t[:, None]
        freqs = torch.exp(
            torch.linspace(0, np.log(10000.0), half, device=t.device, dtype=t.dtype)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


def build_mlp(nn, in_dim: int, hidden_dim: int, out_dim: int):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def make_score_net_class():
    torch, nn = require_torch()

    class LocalPointScoreNet(nn.Module):
        """PointNet-style local score model.

        Args:
            cfg: ScoreNetConfig.

        Forward:
            x: [B, N, 3] noised normalized patch coordinates.
            t: [B] diffusion times in [0, 1].
            returns: [B, N, 3] score vectors in normalized coordinates.
        """

        def __init__(self, cfg: ScoreNetConfig):
            super().__init__()
            self.cfg = cfg
            self.time_embed = SinusoidalTimeEmbedding(cfg.time_dim)
            self.local = build_mlp(nn, 3 + cfg.time_dim, cfg.hidden_dim, cfg.hidden_dim)
            self.out = build_mlp(
                nn,
                cfg.hidden_dim * 2 + cfg.time_dim,
                cfg.hidden_dim,
                3,
            )

        def forward(self, x, t):
            b, n, _ = x.shape
            te = self.time_embed(t)
            te_points = te[:, None, :].expand(b, n, te.shape[-1])
            h = self.local(torch.cat([x, te_points], dim=-1))
            g = h.max(dim=1, keepdim=True).values.expand_as(h)
            return self.out(torch.cat([h, g, te_points], dim=-1))

    return LocalPointScoreNet


def create_score_net(cfg: ScoreNetConfig):
    return make_score_net_class()(cfg)


def save_score_net_checkpoint(path: str | Path, model, cfg: ScoreNetConfig, extra: Dict[str, Any] | None = None):
    torch, _ = require_torch()
    state_model = model.module if hasattr(model, "module") else model
    payload = {
        "format": "asap.local_point_score_net.v1",
        "config": asdict(cfg),
        "state_dict": state_model.state_dict(),
        "extra": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_score_net_checkpoint(path: str | Path, device: str = "cpu"):
    torch, _ = require_torch()
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    cfg = ScoreNetConfig(**payload["config"])
    model = create_score_net(cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, payload.get("extra", {})


def sample_fixed_patch(
    xyz: np.ndarray,
    center: np.ndarray,
    radius: float,
    patch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a fixed-size normalized patch from xyz coordinates."""
    radius = float(max(radius, 1e-4))
    norm = (xyz - center[None, :]) / radius
    if len(norm) == 0:
        norm = np.zeros((1, 3), dtype=np.float32)
    replace = len(norm) < patch_size
    idx = rng.choice(len(norm), size=patch_size, replace=replace)
    return norm[idx].astype(np.float32)
