"""ASAP M3 — Selective purification.

Given a point cloud, a list of SPUs (from M1) and per-SPU anomaly
scores (from M2), return a purified point cloud.

DropPurifier:
    Drop the inner-ball points of every SPU whose score exceeds tau.
    The annulus points of those SPUs and ALL points outside any flagged
    SPU stay unchanged. When an inner point belongs to multiple
    overlapping SPUs (some flagged, some not), it is dropped iff at
    least `policy_min_votes` flagged SPUs include it (default 1).

VPSDEPurifier:
    Same SPU-selective targeting but, instead of dropping, run a
    short-trajectory VP-SDE denoising on the flagged inner points using
    a pretrained score network. Without a trained score net, falls back
    to local-mean smoothing as a stand-in.

HybridPurifier:
    Two-branch ablation that combines `DropPurifier`-style deletion
    with `VPSDEPurifier`-style local-mean smoothing on the same set
    of flagged SPUs.

    Drop branch (cluster):  flagged SPUs whose PCA anisotropy is
    above ``hybrid_anisotropy_threshold`` AND whose score lies above
    a dynamic gate (top ``1 - hybrid_drop_score_quantile`` quantile of
    flagged scores, but at least ``hybrid_score_multiplier * tau``)
    are deleted.

    Drop branch (isolation):  any flagged inner point with fewer than
    ``hybrid_min_neighbors`` spatial neighbors inside
    ``hybrid_isolation_radius`` is also deleted, to handle sparse
    off-surface injected points that local-mean smoothing alone cannot
    dissipate.  Set either parameter to 0 to disable the branch.

    Smooth branch:  every remaining flagged inner point is replaced by
    the 8-NN local mean over non-flagged, non-dropped anchors (same
    fallback as ``VPSDEPurifier``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PurifierConfig:
    """Hyperparameters for selective purification (M3).

    Common:
        tau: decision threshold on the anomaly score.
        policy_min_votes: minimum #flagged SPUs that must contain a point
            for it to be edited.

    VP-SDE specific:
        t_star: truncation time in [0, 1] (fraction of T).
        n_reverse_steps: number of Euler-Maruyama steps in reverse SDE.
        beta_min, beta_max: linear VP-SDE noise schedule.
        score_net: optional callable(x, t) -> score; if None and
            score_net_ckpt is unset, falls back to local-mean smoothing.
    """
    tau: float = 0.5
    policy_min_votes: int = 1

    # VP-SDE
    t_star: float = 0.05
    n_reverse_steps: int = 5
    beta_min: float = 0.1
    beta_max: float = 20.0
    score_net: Optional[object] = None
    score_net_ckpt: Optional[str] = None
    score_net_device: str = "cpu"
    score_step_size: float = 1.0
    score_clip: float = 3.0
    score_batch_spus: int = 64
    score_anchor_blend: float = 0.0
    score_injection_filter_radius: float = 0.0
    score_injection_filter_min_neighbors: int = 0
    score_filter_attack_kinds: str = "injection"

    # Hybrid drop + smooth
    hybrid_anisotropy_threshold: float = 0.97
    hybrid_score_multiplier: float = 2.0
    hybrid_drop_score_quantile: float = 0.90
    hybrid_isolation_radius: float = 0.30
    hybrid_min_neighbors: int = 3
    attack_kind: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_flagged_inner_votes(
    n_points: int,
    spus: List[dict],
    scores: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Count, for every input point, how many *flagged* SPUs include it
    in their inner ball.

    Args:
        n_points: total number of input points.
        spus: list of SPU dicts.
        scores: per-SPU anomaly scores.
        tau: threshold.

    Returns:
        (n_points,) int array of vote counts.
    """
    votes = np.zeros(n_points, dtype=np.int32)
    flagged = scores > tau
    for i, spu in enumerate(spus):
        if not flagged[i]:
            continue
        votes[spu["inner_idx"]] += 1
    return votes


def _collect_selected_inner_votes(
    n_points: int,
    spus: List[dict],
    selected: np.ndarray,
) -> np.ndarray:
    votes = np.zeros(n_points, dtype=np.int32)
    for i, spu in enumerate(spus):
        if not selected[i]:
            continue
        votes[spu["inner_idx"]] += 1
    return votes


# ---------------------------------------------------------------------------
# DropPurifier — the simplest M3 implementation
# ---------------------------------------------------------------------------

class DropPurifier:
    """Selective DROP purifier (no diffusion). The minimal viable M3."""

    def __init__(self, cfg: Optional[PurifierConfig] = None):
        self.cfg = cfg if cfg is not None else PurifierConfig()

    def purify(
        self,
        points: np.ndarray,
        spus: List[dict],
        scores: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Return (purified_points, meta).

        Args:
            points: (N, C) input cloud.
            spus: list of SPU dicts from M1.
            scores: (M,) anomaly scores from M2.

        Returns:
            purified_points: (N', C) cloud with edited / dropped points.
            meta: dict with counts and timing info.
        """
        n = len(points)
        votes = _collect_flagged_inner_votes(n, spus, scores, self.cfg.tau)
        drop_mask = votes >= self.cfg.policy_min_votes
        kept = points[~drop_mask]

        n_flagged_spus = int((scores > self.cfg.tau).sum())
        n_dropped = int(drop_mask.sum())
        meta = {
            "method": "drop",
            "tau": self.cfg.tau,
            "n_spus": len(spus),
            "n_flagged_spus": n_flagged_spus,
            "n_input_points": n,
            "n_kept_points": int(len(kept)),
            "n_dropped_points": n_dropped,
        }
        return kept, meta


# ---------------------------------------------------------------------------
# VPSDEPurifier — paper §3.5 (interface; score-net optional)
# ---------------------------------------------------------------------------

class VPSDEPurifier:
    """VP-SDE selective diffusion (M3 paper §3.5).

    Without a trained score network, falls back to a local-mean smoothing
    of the flagged inner points so the pipeline can be tested end-to-end.
    """

    def __init__(self, cfg: Optional[PurifierConfig] = None):
        self.cfg = cfg if cfg is not None else PurifierConfig()
        self.score_net = self.cfg.score_net
        self.score_net_cfg = None
        if self.score_net is None and self.cfg.score_net_ckpt:
            from asap.diffusion.score_net import load_score_net_checkpoint

            self.score_net, self.score_net_cfg, _ = load_score_net_checkpoint(
                self.cfg.score_net_ckpt,
                device=self.cfg.score_net_device,
            )

    @staticmethod
    def _alpha_bar(beta_min: float, beta_max: float, t: float) -> float:
        """Closed-form survival coefficient for the linear VP schedule.

        beta(s) = beta_min + s * (beta_max - beta_min),
        bar_alpha(t) = exp(-integral_0^t beta(s) ds)
                     = exp(- beta_min * t  - 0.5 * (beta_max - beta_min) * t^2 )
        """
        return float(np.exp(-beta_min * t - 0.5 * (beta_max - beta_min) * t * t))

    def _local_mean_smooth(
        self,
        points: np.ndarray,
        edit_indices: np.ndarray,
        anchor_exclude_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Stand-in for the score network when no model is supplied.

        Replaces each flagged inner point with the mean of its 8 spatial
        neighbors among the *non-flagged* points, i.e. an attractive
        denoising move toward the local manifold.
        """
        from scipy.spatial import cKDTree

        if len(edit_indices) == 0:
            return points.copy()

        all_mask = np.ones(len(points), dtype=bool)
        all_mask[edit_indices] = False
        if anchor_exclude_indices is not None and len(anchor_exclude_indices) > 0:
            all_mask[anchor_exclude_indices] = False
        anchors = points[all_mask, :3]
        if len(anchors) < 8:
            return points.copy()

        tree = cKDTree(anchors)
        _, knn = tree.query(points[edit_indices, :3], k=8, workers=-1)
        smoothed = anchors[knn].mean(axis=1)

        out = points.copy()
        out[edit_indices, :3] = smoothed
        return out

    def _score_net_denoise(
        self,
        points: np.ndarray,
        spus: List[dict],
        scores: np.ndarray,
        edit_mask: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Denoise flagged SPU interiors with the trained local score net.

        The score net is trained in normalized SPU coordinates. At inference
        time each flagged SPU proposes denoised xyz positions for its editable
        inner points; overlapping SPU proposals are averaged per point.
        """
        if self.score_net is None:
            raise RuntimeError("_score_net_denoise called without a score net")

        import torch

        device = next(self.score_net.parameters()).device
        self.score_net.eval()
        out = points.copy()
        xyz_accum = np.zeros((len(points), 3), dtype=np.float64)
        xyz_counts = np.zeros(len(points), dtype=np.int32)
        anchor_blend = float(np.clip(self.cfg.score_anchor_blend, 0.0, 1.0))
        anchor_points = None
        if anchor_blend > 0.0:
            edit_indices = np.where(edit_mask)[0]
            anchor_points = self._local_mean_smooth(points, edit_indices)

        t_value = float(self.cfg.t_star)
        abar = float(np.exp(
            -self.cfg.beta_min * t_value
            - 0.5 * (self.cfg.beta_max - self.cfg.beta_min) * t_value * t_value
        ))
        sigma2 = max(1.0 - abar, 1e-8)
        sqrt_abar = float(np.sqrt(max(abar, 1e-8)))

        n_score_spus = 0
        n_score_points = 0
        flagged = scores > self.cfg.tau
        candidates = []
        for i, spu in enumerate(spus):
            if not flagged[i]:
                continue
            idx = np.asarray(spu["inner_idx"], dtype=np.int64)
            if len(idx) == 0:
                continue
            idx = idx[edit_mask[idx]]
            if len(idx) < 2:
                continue

            radius = float(max(spu["r2"], 1e-4))
            center = np.asarray(spu["center"], dtype=np.float32)
            x0 = (points[idx, :3].astype(np.float32) - center[None, :]) / radius
            x0 = np.clip(x0, -self.cfg.score_clip, self.cfg.score_clip)
            candidates.append((idx, radius, center, x0))

        batch_limit = max(1, int(self.cfg.score_batch_spus))
        batched_world = [None] * len(candidates)
        with torch.no_grad():
            by_size = {}
            for pos, (_, _, _, x0) in enumerate(candidates):
                by_size.setdefault(len(x0), []).append(pos)

            for positions in by_size.values():
                for start in range(0, len(positions), batch_limit):
                    batch_pos = positions[start : start + batch_limit]
                    x0_batch = np.stack(
                        [candidates[pos][3] for pos in batch_pos], axis=0
                    )
                    xt = torch.from_numpy(x0_batch).to(
                        device=device, dtype=torch.float32
                    )
                    xt = xt * sqrt_abar
                    t = torch.full(
                        (len(batch_pos),), t_value, device=device, dtype=torch.float32
                    )
                    pred_score = self.score_net(xt, t)
                    xhat = (xt + sigma2 * pred_score) / sqrt_abar
                    xbase = xt / sqrt_abar
                    xnew = xbase + self.cfg.score_step_size * (xhat - xbase)
                    xnew = torch.clamp(xnew, -self.cfg.score_clip, self.cfg.score_clip)
                    xnew_np = xnew.detach().cpu().numpy()

                    for row, pos in enumerate(batch_pos):
                        _, radius, center, _ = candidates[pos]
                        batched_world[pos] = center[None, :] + radius * xnew_np[row]

        for pos, world in enumerate(batched_world):
            if world is None:
                continue
            idx = candidates[pos][0]
            xyz_accum[idx] += world.astype(np.float64)
            xyz_counts[idx] += 1
            n_score_spus += 1
            n_score_points += len(idx)

        edited = xyz_counts > 0
        if edited.any():
            score_xyz = (xyz_accum[edited] / xyz_counts[edited, None]).astype(np.float32)
            if anchor_points is not None:
                anchor_xyz = anchor_points[edited, :3].astype(np.float32)
                score_xyz = (
                    (1.0 - anchor_blend) * score_xyz
                    + anchor_blend * anchor_xyz
                ).astype(np.float32)
            out[edited, :3] = score_xyz

        meta = {
            "score_net_ckpt": self.cfg.score_net_ckpt,
            "score_net_device": self.cfg.score_net_device,
            "score_step_size": self.cfg.score_step_size,
            "score_clip": self.cfg.score_clip,
            "score_batch_spus": int(self.cfg.score_batch_spus),
            "score_anchor_blend": anchor_blend,
            "n_score_spus": int(n_score_spus),
            "n_score_point_proposals": int(n_score_points),
            "n_score_edited_points": int(edited.sum()),
        }
        return out, meta

    def _apply_score_injection_filter(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Optional Injection-specific radius filter after score denoising.

        This keeps the Track A score-net path as the main purifier while
        allowing a conservative removal branch for off-surface points, which
        pure local smoothing tends to collapse rather than remove. By default
        the branch is enabled only for attack_kind == "injection"; diagnostics
        can opt in additional attack kinds via score_filter_attack_kinds.
        """
        radius = float(self.cfg.score_injection_filter_radius)
        min_neighbors = int(self.cfg.score_injection_filter_min_neighbors)
        allowed_kinds = {
            item.strip()
            for item in str(self.cfg.score_filter_attack_kinds).split(",")
            if item.strip()
        }
        enabled = (
            self.cfg.attack_kind in allowed_kinds
            and radius > 0.0
            and min_neighbors > 0
        )
        meta = {
            "score_injection_filter_radius": radius,
            "score_injection_filter_min_neighbors": min_neighbors,
            "score_filter_attack_kinds": sorted(allowed_kinds),
            "score_injection_filter_enabled": bool(enabled),
            "n_score_injection_filter_dropped_points": 0,
        }
        if not enabled or len(points) == 0:
            return points, meta

        from scipy.spatial import cKDTree

        tree = cKDTree(points[:, :3].astype(np.float64))
        counts = np.array(
            tree.query_ball_point(
                points[:, :3],
                radius,
                workers=-1,
                return_length=True,
            )
        ) - 1
        keep_mask = counts >= min_neighbors
        meta["n_score_injection_filter_dropped_points"] = int((~keep_mask).sum())
        return points[keep_mask], meta

    def purify(
        self,
        points: np.ndarray,
        spus: List[dict],
        scores: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        n = len(points)
        votes = _collect_flagged_inner_votes(n, spus, scores, self.cfg.tau)
        edit_mask = votes >= self.cfg.policy_min_votes
        edit_indices = np.where(edit_mask)[0]

        if self.score_net is None:
            out = self._local_mean_smooth(points, edit_indices)
            method = "local_mean_fallback"
            score_meta = {}
        else:
            out, score_meta = self._score_net_denoise(points, spus, scores, edit_mask)
            method = "vp_sde_score_net"

        out, injection_filter_meta = self._apply_score_injection_filter(out)

        meta = {
            "method": method,
            "tau": self.cfg.tau,
            "t_star": self.cfg.t_star,
            "n_spus": len(spus),
            "n_flagged_spus": int((scores > self.cfg.tau).sum()),
            "n_input_points": n,
            "n_kept_points": int(len(out)),
            "n_edited_points": int(len(edit_indices)),
            **score_meta,
            **injection_filter_meta,
        }
        return out, meta


class HybridPurifier:
    def __init__(self, cfg: Optional[PurifierConfig] = None):
        self.cfg = cfg if cfg is not None else PurifierConfig()

    def purify(
        self,
        points: np.ndarray,
        spus: List[dict],
        scores: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        from asap.scoring import compute_features

        n = len(points)
        flagged = scores > self.cfg.tau
        n_flagged_spus = int(flagged.sum())

        if n_flagged_spus == 0:
            meta = {
                "method": "hybrid",
                "tau": self.cfg.tau,
                "n_spus": len(spus),
                "n_flagged_spus": 0,
                "n_drop_spus": 0,
                "n_smooth_spus": 0,
                "n_input_points": n,
                "n_kept_points": n,
                "n_dropped_points": 0,
                "n_edited_points": 0,
            }
            return points.copy(), meta

        features = compute_features(points, spus)
        anisotropy = features[:, 1]
        flagged_scores = scores[flagged]
        dynamic_gate = float(np.quantile(flagged_scores, self.cfg.hybrid_drop_score_quantile))
        absolute_gate = float(self.cfg.tau * self.cfg.hybrid_score_multiplier)
        score_gate = scores >= max(dynamic_gate, absolute_gate)

        drop_spus = (
            flagged
            & (anisotropy >= self.cfg.hybrid_anisotropy_threshold)
            & score_gate
        )
        smooth_spus = flagged & ~drop_spus

        drop_votes = _collect_selected_inner_votes(n, spus, drop_spus)
        smooth_votes = _collect_selected_inner_votes(n, spus, smooth_spus)
        flagged_votes = _collect_selected_inner_votes(n, spus, flagged)

        drop_mask = drop_votes >= self.cfg.policy_min_votes
        use_isolation_drop = self.cfg.attack_kind in (None, "injection")
        if (
            use_isolation_drop
            and self.cfg.hybrid_min_neighbors > 0
            and self.cfg.hybrid_isolation_radius > 0
        ):
            from scipy.spatial import cKDTree

            tree = cKDTree(points[:, :3])
            neighbor_counts = np.array(
                tree.query_ball_point(
                    points[:, :3],
                    self.cfg.hybrid_isolation_radius,
                    workers=-1,
                    return_length=True,
                )
            ) - 1
            isolated_mask = (
                (flagged_votes >= self.cfg.policy_min_votes)
                & (neighbor_counts < self.cfg.hybrid_min_neighbors)
            )
            drop_mask = drop_mask | isolated_mask
        else:
            isolated_mask = np.zeros(n, dtype=bool)

        smooth_mask = (smooth_votes >= self.cfg.policy_min_votes) & ~drop_mask

        drop_indices = np.where(drop_mask)[0]
        smooth_indices = np.where(smooth_mask)[0]

        smoothed = VPSDEPurifier(self.cfg)._local_mean_smooth(
            points,
            smooth_indices,
            anchor_exclude_indices=drop_indices,
        )
        kept = smoothed[~drop_mask]

        meta = {
            "method": "hybrid",
            "tau": self.cfg.tau,
            "hybrid_anisotropy_threshold": self.cfg.hybrid_anisotropy_threshold,
            "hybrid_score_multiplier": self.cfg.hybrid_score_multiplier,
            "hybrid_drop_score_quantile": self.cfg.hybrid_drop_score_quantile,
            "hybrid_isolation_radius": self.cfg.hybrid_isolation_radius,
            "hybrid_min_neighbors": self.cfg.hybrid_min_neighbors,
            "attack_kind": self.cfg.attack_kind,
            "hybrid_use_isolation_drop": use_isolation_drop,
            "hybrid_dynamic_score_gate": dynamic_gate,
            "hybrid_absolute_score_gate": absolute_gate,
            "n_spus": len(spus),
            "n_flagged_spus": n_flagged_spus,
            "n_drop_spus": int(drop_spus.sum()),
            "n_smooth_spus": int(smooth_spus.sum()),
            "n_input_points": n,
            "n_kept_points": int(len(kept)),
            "n_dropped_points": int(len(drop_indices)),
            "n_isolated_dropped_points": int(isolated_mask.sum()),
            "n_edited_points": int(len(smooth_indices)),
        }
        return kept, meta


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_purifier(
    method: str,
    points: np.ndarray,
    spus: List[dict],
    scores: np.ndarray,
    cfg: Optional[PurifierConfig] = None,
) -> Tuple[np.ndarray, dict]:
    """Convenience dispatcher: method in {'drop', 'vp_sde', 'hybrid'}."""
    if method == "drop":
        return DropPurifier(cfg).purify(points, spus, scores)
    elif method == "vp_sde":
        return VPSDEPurifier(cfg).purify(points, spus, scores)
    elif method == "hybrid":
        return HybridPurifier(cfg).purify(points, spus, scores)
    else:
        raise ValueError(f"Unknown purifier method: {method}")
