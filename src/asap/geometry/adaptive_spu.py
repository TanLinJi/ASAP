"""ASAP M1 — Adaptive-Radius Spherical Purification Unit (SPU) construction.

Implements the algorithm specified in docs/paper/03_method.md §3.3:

  1. Pick candidate SPU centers via Farthest Point Sampling (FPS) at
     stride eta from the full input cloud P.
  2. For each center p, compute d_k(p) = distance to its k-th nearest
     neighbor in P \\ {p}.
  3. Set outer radius:   r1(p) = clip(alpha * d_k(p), [r_min, r_max]).
  4. Set inner radius:   r2(p) = beta * r1(p),   with beta in (0,1).
  5. Define
        S(p)     = points in P within  r1(p) of p,
        P_in(p)  = points in P within  r2(p) of p,
        P_an(p)  = S(p) \\ P_in(p)    (annulus, M2-only).

The annulus stabilises the M2 statistics; only P_in is editable by M3.

Defaults from paper §3.3.4:
    alpha = 2,  k = 16,  beta = 0.67,
    r_min = 0.10 m,  r_max = 0.40 m.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SPUConfig:
    """Hyperparameters for Adaptive-Radius SPU construction (M1).

    Defaults match docs/paper/03_method.md §3.3.4.
    """
    k: int = 16
    """Neighbor count used for the local-scale estimator d_k(p)."""

    alpha: float = 2.0
    """Multiplier on d_k(p) for the outer radius r_1(p)."""

    beta: float = 0.67
    """Multiplier in (0,1) for the inner radius r_2(p) = beta * r_1(p)."""

    r_min: float = 0.10
    """Lower clip on r_1(p) in meters (prevents tiny SPUs in dense regions)."""

    r_max: float = 0.40
    """Upper clip on r_1(p) in meters (prevents huge SPUs in sparse regions)."""

    eta: float = 0.30
    """FPS sampling stride in meters used to pick candidate centers.

    Smaller eta => more SPUs => better coverage but more compute.
    Setting eta ~ r_min keeps overlap modest while ensuring an r_1-cover.
    """

    max_centers: Optional[int] = None
    """Optional cap on the number of FPS centers (None = unbounded)."""

    min_points_per_spu: int = 4
    """SPUs with fewer than this many points in S(p) are dropped."""

    backend: str = "cpu"
    """M1 backend. `cpu` uses SciPy cKDTree; `torch` uses torch distance
    kernels for kNN/radius queries while keeping the same center sampler."""

    torch_device: str = "cuda"
    """Device for the experimental torch backend."""

    torch_chunk_centers: int = 512
    """Center chunk size for torch radius queries."""


# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------

def knn_distances(xyz: np.ndarray, k: int) -> np.ndarray:
    """Return d_k(p) for every point in xyz.

    Args:
        xyz: (N, 3) float array of point coordinates.
        k: 1-based neighbor index (the k-th nearest neighbor, excluding self).

    Returns:
        (N,) float array of distances to the k-th nearest neighbor.
    """
    if len(xyz) <= k:
        # Not enough points; pad with the largest available distance.
        if len(xyz) <= 1:
            return np.zeros(len(xyz), dtype=np.float64)
        tree = cKDTree(xyz)
        dists, _ = tree.query(xyz, k=len(xyz), workers=-1)
        return dists[:, -1].astype(np.float64)

    tree = cKDTree(xyz)
    # k+1 because the first neighbor is the point itself
    dists, _ = tree.query(xyz, k=k + 1, workers=-1)
    return dists[:, -1].astype(np.float64)


def farthest_point_sampling(
    xyz: np.ndarray,
    stride: float,
    max_points: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """KD-tree stride-net: a fast surrogate for FPS at large N.

    Greedy stride-net algorithm (also called "Poisson-disk" or "Bridson"
    sampling for fixed radius):

      1. Build a KD-tree over xyz.
      2. Maintain a boolean `available` mask, all True initially.
      3. Pick a random available point, add to selected, mark all
         available points within `stride` as unavailable (KD-tree ball query).
      4. Repeat until no available points remain or max_points reached.

    Properties: every selected center is >= stride from every other, and
    every input point is within `stride` of some center — i.e. the
    selected set is a stride-cover of xyz, which is exactly the property
    FPS provides for ASAP M1 §3.3.3.

    Complexity: O(N log N + total_ball_query_work), which is dramatically
    faster than O(N * M) classical FPS when N >> M.

    Args:
        xyz: (N, 3) point coordinates.
        stride: Minimum spatial separation between selected centers (m).
        max_points: Optional upper bound on number of returned centers.
        seed: RNG seed for picking the initial point and tie-breaks.

    Returns:
        (M,) int array of indices into xyz; M <= N (and <= max_points).
    """
    n = len(xyz)
    if n == 0:
        return np.array([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    tree = cKDTree(xyz)

    available = np.ones(n, dtype=bool)
    selected: List[int] = []
    cap = max_points if max_points is not None else n

    # Random order over points: pick available points in this order.
    order = rng.permutation(n)

    for idx in order:
        if not available[idx]:
            continue
        selected.append(int(idx))
        if len(selected) >= cap:
            break
        # Mark all available points within `stride` as unavailable.
        nbrs = tree.query_ball_point(xyz[idx], r=stride, workers=-1)
        if nbrs:
            available[nbrs] = False

    return np.asarray(selected, dtype=np.int64)


# ---------------------------------------------------------------------------
# SPU builder
# ---------------------------------------------------------------------------

def build_spus(
    points: np.ndarray,
    cfg: Optional[SPUConfig] = None,
    seed: int = 0,
) -> List[dict]:
    """Build a list of Adaptive-Radius SPUs from a raw point cloud.

    Args:
        points: (N, C) input point cloud; only the first 3 cols are used
            for geometry, extra columns (reflectance, etc.) are preserved.
        cfg: SPUConfig; if None, paper defaults are used.
        seed: RNG seed (passed to FPS).

    Returns:
        List of dicts, one per SPU, with keys:
            center_idx:  int   — index of the center in `points`
            center:      (3,)  — xyz of the center
            r1:          float — outer radius (m)
            r2:          float — inner radius (m)
            sphere_idx:  (Ks,) — indices of all points in S(p)
            inner_idx:   (Ki,) — indices of points in P_in(p)
            annulus_idx: (Ka,) — indices of points in P_an(p)

    Notes:
        The same point index can appear in multiple SPUs (overlapping balls).
        Downstream code is responsible for collision policy at the pipeline
        level (e.g., union of in-mask for selective diffusion).
    """
    if cfg is None:
        cfg = SPUConfig()
    xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    if len(xyz) == 0:
        return []

    # 1) FPS candidate centers
    center_indices = farthest_point_sampling(
        xyz, stride=cfg.eta, max_points=cfg.max_centers, seed=seed
    )

    # 2) k-NN distances over the WHOLE cloud once (cheap with cKDTree).
    tree = cKDTree(xyz)
    k_eff = min(cfg.k, len(xyz) - 1) if len(xyz) > 1 else 1
    dists, _ = tree.query(xyz[center_indices], k=k_eff + 1, workers=-1)
    d_k = dists[:, -1].astype(np.float64)  # k-th NN distance (k=last col)

    # 3) Adaptive outer radius r1, inner radius r2
    r1 = np.clip(cfg.alpha * d_k, cfg.r_min, cfg.r_max)
    r2 = cfg.beta * r1

    # 4) Ball queries to define S, P_in, P_an
    # query_ball_point with a *list* of radii returns per-point lists.
    sphere_idx_lists = tree.query_ball_point(
        xyz[center_indices], r=r1, workers=-1
    )

    spus: List[dict] = []
    for i, c_idx in enumerate(center_indices):
        s_idx = np.asarray(sphere_idx_lists[i], dtype=np.int64)
        if len(s_idx) < cfg.min_points_per_spu:
            continue

        # Vectorised inner / annulus split using current SPU radii.
        c = xyz[c_idx]
        diff = xyz[s_idx] - c
        d2 = np.einsum("ij,ij->i", diff, diff)
        inner_mask = d2 <= (r2[i] * r2[i])
        inner_idx = s_idx[inner_mask]
        annulus_idx = s_idx[~inner_mask]

        spus.append({
            "center_idx": int(c_idx),
            "center": c.astype(np.float32),
            "r1": float(r1[i]),
            "r2": float(r2[i]),
            "sphere_idx": s_idx,
            "inner_idx": inner_idx,
            "annulus_idx": annulus_idx,
        })

    return spus


def build_spus_torch(
    points: np.ndarray,
    cfg: Optional[SPUConfig] = None,
    seed: int = 0,
) -> List[dict]:
    """Experimental torch-backed M1 builder.

    The center set is intentionally kept identical to the CPU stride-net
    sampler. kNN local scale and ball queries are computed with torch in
    chunks, which can run on GPU and reduces SciPy KD-tree pressure.
    """
    if cfg is None:
        cfg = SPUConfig()
    xyz_np = np.ascontiguousarray(points[:, :3], dtype=np.float32)
    if len(xyz_np) == 0:
        return []

    center_indices = farthest_point_sampling(
        xyz_np.astype(np.float64),
        stride=cfg.eta,
        max_points=cfg.max_centers,
        seed=seed,
    )
    if len(center_indices) == 0:
        return []

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError("SPUConfig.backend='torch' requires PyTorch") from exc

    device = torch.device(
        cfg.torch_device if torch.cuda.is_available() or cfg.torch_device == "cpu" else "cpu"
    )
    xyz = torch.as_tensor(xyz_np, device=device, dtype=torch.float32)
    centers = xyz[torch.as_tensor(center_indices, device=device, dtype=torch.long)]

    k_eff = min(cfg.k, len(xyz_np) - 1) if len(xyz_np) > 1 else 1
    if len(xyz_np) <= 1:
        d_k = np.zeros(len(center_indices), dtype=np.float64)
    else:
        d_k_parts = []
        chunk = max(1, int(cfg.torch_chunk_centers))
        for start in range(0, len(center_indices), chunk):
            c = centers[start : start + chunk]
            dist = torch.cdist(c, xyz)
            _, topi = torch.topk(dist, k=k_eff + 1, largest=False, dim=1)
            gathered = torch.gather(dist, 1, topi)
            d_k_parts.append(gathered[:, -1].detach().cpu())
            del dist, topi, gathered
        d_k = torch.cat(d_k_parts).numpy().astype(np.float64)

    r1 = np.clip(cfg.alpha * d_k, cfg.r_min, cfg.r_max).astype(np.float32)
    r2 = (cfg.beta * r1).astype(np.float32)

    spus: List[dict] = []
    chunk = max(1, int(cfg.torch_chunk_centers))
    with torch.no_grad():
        for start in range(0, len(center_indices), chunk):
            end = min(start + chunk, len(center_indices))
            c = centers[start:end]
            dist2 = torch.cdist(c, xyz).square()
            r1_t = torch.as_tensor(r1[start:end], device=device, dtype=torch.float32)
            r2_t = torch.as_tensor(r2[start:end], device=device, dtype=torch.float32)
            sphere_mask = dist2 <= (r1_t[:, None] * r1_t[:, None])
            inner_mask = dist2 <= (r2_t[:, None] * r2_t[:, None])
            sphere_lists = [torch.where(row)[0].detach().cpu().numpy() for row in sphere_mask]
            inner_lists = [torch.where(row)[0].detach().cpu().numpy() for row in inner_mask]
            del dist2, sphere_mask, inner_mask

            for j, c_idx in enumerate(center_indices[start:end]):
                s_idx = sphere_lists[j].astype(np.int64, copy=False)
                if len(s_idx) < cfg.min_points_per_spu:
                    continue
                inner_idx = inner_lists[j].astype(np.int64, copy=False)
                if len(inner_idx) == 0:
                    annulus_idx = s_idx
                else:
                    annulus_idx = np.setdiff1d(s_idx, inner_idx, assume_unique=False)
                c_np = xyz_np[int(c_idx)]
                spus.append({
                    "center_idx": int(c_idx),
                    "center": c_np.astype(np.float32),
                    "r1": float(r1[start + j]),
                    "r2": float(r2[start + j]),
                    "sphere_idx": s_idx,
                    "inner_idx": inner_idx,
                    "annulus_idx": annulus_idx,
                })

    return spus


# ---------------------------------------------------------------------------
# OO wrapper for convenience
# ---------------------------------------------------------------------------

class AdaptiveSPUBuilder:
    """Thin OO wrapper around :func:`build_spus`.

    Example:
        >>> import numpy as np
        >>> pts = np.fromfile("000001.bin", dtype=np.float32).reshape(-1, 4)
        >>> builder = AdaptiveSPUBuilder()  # paper defaults
        >>> spus = builder.build(pts)
        >>> print(f"{len(spus)} SPUs constructed")
    """

    def __init__(self, cfg: Optional[SPUConfig] = None):
        self.cfg = cfg if cfg is not None else SPUConfig()

    def build(self, points: np.ndarray, seed: int = 0) -> List[dict]:
        if self.cfg.backend == "torch":
            return build_spus_torch(points, self.cfg, seed=seed)
        if self.cfg.backend != "cpu":
            raise ValueError(f"Unknown SPU backend: {self.cfg.backend}")
        return build_spus(points, self.cfg, seed=seed)
