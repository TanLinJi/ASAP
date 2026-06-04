"""ASAP M3 — Selective Diffusion (and gentler editing) module.

Two implementations are provided:

- :class:`DropPurifier` — the simplest selective editor: for every SPU
  whose anomaly score s(p) > tau, drop the *inner* points P_in(p). The
  annulus and all unselected SPUs remain untouched. This is intended as
  an immediate end-to-end pipeline so that the ASAP method can be
  evaluated against detectors without waiting for a fully trained
  score network.

- :class:`VPSDEPurifier` — VP-SDE selective diffusion (paper §3.5).
  Pretrained score network supported through `score_net.py`; local-mean
  fallback remains available for ablation and debugging.
"""

from .score_net import (
    ScoreNetConfig,
    create_score_net,
    load_score_net_checkpoint,
    save_score_net_checkpoint,
)
from .selective_purifier import (
    DropPurifier,
    HybridPurifier,
    PurifierConfig,
    VPSDEPurifier,
    apply_purifier,
)

__all__ = [
    "DropPurifier",
    "HybridPurifier",
    "PurifierConfig",
    "ScoreNetConfig",
    "VPSDEPurifier",
    "apply_purifier",
    "create_score_net",
    "load_score_net_checkpoint",
    "save_score_net_checkpoint",
]
