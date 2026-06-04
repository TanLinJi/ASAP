"""ASAP M2 — Anomaly Scorer for Spherical Purification Units.

Implements §3.4 of docs/paper/03_method.md:

  f_c : compactness     = |P_in| / V_2                    (M2.1)
  f_a : PCA anisotropy  = 1 - lambda_3 / (l1+l2+l3)       (M2.2)
  f_v : vMF concentration kappa-hat (Banerjee 2005)       (M2.3)
  f_d : density ratio   = (|P_in|/V_2) / (|P_an|/V_an+eps)(M2.4)

Aggregation:
  s(p) = sigmoid( w_c * f_c~ + w_a * f_a~ + w_v * f_v~ + w_d * f_d~ + b )

Per-feature z-normalization (f_*~) uses statistics estimated on a
clean reference set (calibration). Defaults are mild and intended as
sane out-of-the-box weights; the formal calibration procedure (logistic
regression on labeled SPUs + Youden-J on ROC) is provided as a separate
calibration utility (calibrate_scorer).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_FOUR_THIRDS_PI = 4.0 / 3.0 * np.pi
_EPS = 1e-8


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScorerConfig:
    """Hyperparameters for the M2 anomaly scorer.

    The weights default to a uniform positive contribution (no field is
    favoured a priori). A proper calibration step (`calibrate_scorer`)
    fits both the per-feature z-stats and the (w, b) weights from labeled
    clean / attacked SPUs.

    All four normalised features are oriented so that *larger* values are
    *more* anomalous (compactness, anisotropy, kappa, density-ratio all
    grow under cluster / planar / spray / density-spike attacks).
    """
    # Per-feature mean / std used for z-normalisation. Calibrated.
    mu_c: float = 0.0
    sd_c: float = 1.0
    mu_a: float = 0.0
    sd_a: float = 1.0
    mu_v: float = 0.0
    sd_v: float = 1.0
    mu_d: float = 0.0
    sd_d: float = 1.0

    # Logistic aggregator weights.
    w_c: float = 1.0
    w_a: float = 1.0
    w_v: float = 1.0
    w_d: float = 1.0
    b:   float = -2.0  # negative bias so default-uncalibrated SPUs score < 0.5

    # Decision threshold (Youden-J optimal). Default 0.5 until calibrated.
    tau: float = 0.5

    # Attack-type prior inferred during calibration. One of
    # {"injection", "perturbation", "dropping", None}. Used by downstream
    # M3 (e.g. HybridPurifier) to switch attack-specific branches on/off.
    # None preserves the pre-existing behaviour (no attack-conditional gating).
    attack_kind: Optional[str] = None

    @classmethod
    def from_json(cls, path: str | Path) -> "ScorerConfig":
        with open(path) as f:
            d = json.load(f)
        # Backward compatibility: drop unknown keys so older JSONs without
        # newer fields still load, and newer JSONs with future fields don't
        # crash older code.
        valid = {fld.name for fld in fields(cls)}
        d = {k: v for k, v in d.items() if k in valid}
        return cls(**d)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_KAPPA_CAP = 1e3   # cap before log1p; saturates around log1p(1e3) ~ 6.9
_DENS_CAP  = 1e3   # cap for density ratio f_d


def _vmf_kappa_hat(unit_dirs: np.ndarray) -> float:
    """Closed-form vMF concentration estimator (Banerjee 2005), d=3.

    kappa_hat = R * (d - R^2) / (1 - R^2),   R = ||mean(unit_dirs)||.

    Args:
        unit_dirs: (m, 3) unit vectors on S^2 (assumed already normalised).

    Returns:
        Scalar kappa-hat in [0, _KAPPA_CAP], capped to keep the downstream
        log1p / linear aggregator numerically stable.
    """
    m = len(unit_dirs)
    if m == 0:
        return 0.0
    u_bar = unit_dirs.mean(axis=0)
    R = float(np.linalg.norm(u_bar))
    R2 = R * R
    if R2 >= 1.0 - _EPS:
        return _KAPPA_CAP
    val = R * (3.0 - R2) / (1.0 - R2)
    return min(val, _KAPPA_CAP)


def compute_features(
    points: np.ndarray,
    spus: List[dict],
) -> np.ndarray:
    """Compute the 4 raw anomaly features for every SPU.

    Args:
        points: (N, C) full point cloud (only first 3 cols used).
        spus: list of SPU dicts from `asap.geometry.build_spus`.

    Returns:
        (M, 4) float array of [f_c, f_a, f_v, f_d] per SPU.
    """
    xyz = points[:, :3].astype(np.float64)
    feats = np.zeros((len(spus), 4), dtype=np.float64)
    groups: Dict[int, list[int]] = defaultdict(list)

    for i, spu in enumerate(spus):
        r1 = spu["r1"]
        r2 = spu["r2"]
        inner_idx = spu["inner_idx"]
        annulus_idx = spu["annulus_idx"]

        n_in = len(inner_idx)
        n_an = len(annulus_idx)

        v2 = _FOUR_THIRDS_PI * (r2 ** 3)
        v1 = _FOUR_THIRDS_PI * (r1 ** 3)
        v_an = v1 - v2

        # --- f_c: compactness ----------------------------------------
        f_c = n_in / max(v2, _EPS)

        # --- f_d: density ratio (capped) ------------------------------
        rho_in = n_in / max(v2, _EPS)
        rho_an = n_an / max(v_an, _EPS) if v_an > _EPS else 0.0
        # When annulus is empty but inner is non-empty: cap at _DENS_CAP.
        f_d = min(rho_in / (rho_an + _EPS), _DENS_CAP)

        feats[i, 0] = f_c
        feats[i, 3] = f_d
        if n_in >= 2:
            groups[n_in].append(i)

    for n_in, ids in groups.items():
        if not ids:
            continue
        # Group by identical point count so P_in has a dense [B, n, 3] shape.
        idx = np.stack([spus[i]["inner_idx"] for i in ids], axis=0).astype(np.int64)
        p_in = xyz[idx]
        centers = xyz[[spus[i]["center_idx"] for i in ids]]

        # --- f_v: vMF kappa -------------------------------------------
        d_vec = p_in - centers[:, None, :]
        norms = np.linalg.norm(d_vec, axis=2)
        valid = norms > _EPS
        unit = np.zeros_like(d_vec)
        np.divide(d_vec, norms[:, :, None], out=unit, where=valid[:, :, None])
        valid_counts = valid.sum(axis=1)
        u_bar = unit.sum(axis=1)
        np.divide(
            u_bar,
            valid_counts[:, None],
            out=u_bar,
            where=valid_counts[:, None] > 0,
        )
        r = np.linalg.norm(u_bar, axis=1)
        r2 = r * r
        kappa = np.zeros(len(ids), dtype=np.float64)
        ok = (valid_counts >= 2) & (r2 < 1.0 - _EPS)
        kappa[ok] = r[ok] * (3.0 - r2[ok]) / (1.0 - r2[ok])
        kappa[(valid_counts >= 2) & ~ok] = _KAPPA_CAP
        feats[ids, 2] = np.minimum(kappa, _KAPPA_CAP)

        # --- f_a: PCA anisotropy --------------------------------------
        if n_in >= 3:
            centered = p_in - p_in.mean(axis=1, keepdims=True)
            cov = np.einsum("bni,bnj->bij", centered, centered) / n_in
            eigs = np.linalg.eigvalsh(cov)
            eigs = np.clip(eigs, 0.0, None)
            denom = eigs.sum(axis=1)
            anis = np.zeros(len(ids), dtype=np.float64)
            np.divide(eigs[:, 0], denom, out=anis, where=denom > _EPS)
            feats[ids, 1] = np.where(denom > _EPS, 1.0 - anis, 0.0)

    # log1p transform for heavy-tailed features (f_c, f_v, f_d) so the
    # linear aggregator and z-normalisation behave well.
    feats[:, 0] = np.log1p(np.clip(feats[:, 0], 0.0, None))  # f_c
    feats[:, 2] = np.log1p(np.clip(feats[:, 2], 0.0, None))  # f_v
    feats[:, 3] = np.log1p(np.clip(feats[:, 3], 0.0, None))  # f_d
    return feats


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def score_spus(
    features: np.ndarray,
    cfg: ScorerConfig,
) -> np.ndarray:
    """Aggregate per-SPU features into anomaly scores in [0, 1].

    Args:
        features: (M, 4) array of raw [f_c, f_a, f_v, f_d] features.
        cfg: ScorerConfig with z-stats and logistic weights.

    Returns:
        (M,) float array of anomaly scores in [0, 1].
    """
    mu = np.array([cfg.mu_c, cfg.mu_a, cfg.mu_v, cfg.mu_d], dtype=np.float64)
    sd = np.array([cfg.sd_c, cfg.sd_a, cfg.sd_v, cfg.sd_d], dtype=np.float64)
    w  = np.array([cfg.w_c,  cfg.w_a,  cfg.w_v,  cfg.w_d],  dtype=np.float64)

    sd_safe = np.where(sd > _EPS, sd, 1.0)
    z = (features - mu) / sd_safe
    logits = z @ w + cfg.b
    # Clamp to avoid overflow for extreme kappa values
    logits = np.clip(logits, -50.0, 50.0)
    return _sigmoid(logits)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_scorer(
    features_clean: np.ndarray,
    features_attacked: np.ndarray,
    cfg_init: Optional[ScorerConfig] = None,
    youden_threshold: bool = True,
) -> ScorerConfig:
    """Fit z-stats + logistic weights from labeled clean / attacked SPUs.

    Args:
        features_clean: (M0, 4) features from clean SPUs (label=0).
        features_attacked: (M1, 4) features from attacked SPUs (label=1).
        cfg_init: starting config (only used for unset defaults if no fit).
        youden_threshold: if True, set cfg.tau to the Youden-J optimal.

    Returns:
        Fitted ScorerConfig.

    Notes:
        - z-stats (mu, sd) are computed on the *clean* set so that a
          clean SPU has feature ~N(0, 1) and an anomalous SPU shows up
          as a large positive deviation.
        - Logistic weights are fit via scipy logistic regression
          (sklearn is heavier; we use scipy.optimize for a tiny problem).
        - tau is the Youden-J argmax on the ROC traced over both classes.
    """
    from scipy.optimize import minimize

    # Robust z-stats from clean.
    mu = features_clean.mean(axis=0)
    sd = features_clean.std(axis=0)
    sd = np.where(sd > _EPS, sd, 1.0)

    z_clean = (features_clean - mu) / sd
    z_atk = (features_attacked - mu) / sd
    z_all = np.concatenate([z_clean, z_atk], axis=0)
    y_all = np.concatenate([np.zeros(len(z_clean)), np.ones(len(z_atk))]).astype(np.float64)

    # Logistic regression by L-BFGS-B on negative log-likelihood.
    def nll(theta):
        w_local = theta[:4]
        b_local = theta[4]
        logits = z_all @ w_local + b_local
        logits = np.clip(logits, -50.0, 50.0)
        # log(1+exp(-y_tilde * z))  where y_tilde in {-1,+1}
        y_tilde = 2.0 * y_all - 1.0
        z = -y_tilde * logits
        # log1p(exp(z)) numerically stable
        return float(np.mean(np.log1p(np.exp(np.minimum(z, 50.0))) + np.maximum(z - 50.0, 0.0)))

    theta0 = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    res = minimize(nll, theta0, method="L-BFGS-B")
    w_fit = res.x[:4]
    b_fit = float(res.x[4])

    cfg = cfg_init if cfg_init is not None else ScorerConfig()
    cfg = ScorerConfig(
        mu_c=mu[0], sd_c=sd[0],
        mu_a=mu[1], sd_a=sd[1],
        mu_v=mu[2], sd_v=sd[2],
        mu_d=mu[3], sd_d=sd[3],
        w_c=float(w_fit[0]), w_a=float(w_fit[1]),
        w_v=float(w_fit[2]), w_d=float(w_fit[3]),
        b=b_fit, tau=cfg_init.tau if cfg_init else 0.5,
    )

    if youden_threshold:
        scores = score_spus(np.concatenate([features_clean, features_attacked]), cfg)
        labels = y_all
        # Sweep thresholds and find Youden-J argmax.
        ts = np.linspace(0.0, 1.0, 1001)
        best_j = -1.0
        best_t = 0.5
        for t in ts:
            pred = scores > t
            tp = ((pred == 1) & (labels == 1)).sum()
            fn = ((pred == 0) & (labels == 1)).sum()
            fp = ((pred == 1) & (labels == 0)).sum()
            tn = ((pred == 0) & (labels == 0)).sum()
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            j = tpr - fpr
            if j > best_j:
                best_j = j
                best_t = float(t)
        cfg.tau = best_t
        logger.info(f"Youden-J optimal tau={best_t:.3f} (J={best_j:.3f})")

    return cfg


# ---------------------------------------------------------------------------
# OO wrapper
# ---------------------------------------------------------------------------

class AnomalyScorer:
    """Thin wrapper combining feature extraction and scoring."""

    def __init__(self, cfg: Optional[ScorerConfig] = None):
        self.cfg = cfg if cfg is not None else ScorerConfig()

    def score(
        self,
        points: np.ndarray,
        spus: List[dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (features, scores) for every SPU."""
        features = compute_features(points, spus)
        scores = score_spus(features, self.cfg)
        return features, scores
