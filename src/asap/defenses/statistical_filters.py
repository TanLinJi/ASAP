"""E3.1 / E3.2 — Statistical Outlier Removal (SOR) and Radius Outlier Removal (ROR).

These are the classical PCL-style point-cloud denoising baselines used
as non-learning, detector-agnostic defense references in the ASAP paper.

Both filters operate on the xyz coordinates and preserve reflectance
for the retained points.

Usage:
    python -m asap.defenses.statistical_filters \\
        --method sor \\
        --input_dir outputs/attacks/kitti/E2.1_injection/velodyne_attacked \\
        --out_dir outputs/defenses/kitti/E3.1_sor/E2.1_injection/velodyne_defended \\
        --k_neighbors 20 --alpha 1.0 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SOR — Statistical Outlier Removal
# ---------------------------------------------------------------------------

def sor_filter(
    pts: np.ndarray,
    k: int = 20,
    alpha: float = 1.0,
) -> np.ndarray:
    """Statistical Outlier Removal.

    For each point, compute the mean distance to its k nearest neighbors.
    A point is kept if its mean-kNN-distance is within (global_mean + alpha * global_std).

    Args:
        pts: (N, C) point cloud with C >= 3; only first 3 cols used for filtering.
        k: Number of neighbors for kNN.
        alpha: Multiplier on standard deviation for the threshold.

    Returns:
        Boolean mask (N,) — True for inliers.
    """
    xyz = pts[:, :3].astype(np.float64)
    tree = cKDTree(xyz)

    # k+1 because query includes the point itself
    dists, _ = tree.query(xyz, k=k + 1, workers=-1)
    mean_dists = dists[:, 1:].mean(axis=1)  # exclude self-distance

    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + alpha * global_std

    inlier_mask = mean_dists <= threshold
    return inlier_mask


# ---------------------------------------------------------------------------
# ROR — Radius Outlier Removal
# ---------------------------------------------------------------------------

def ror_filter(
    pts: np.ndarray,
    radius: float = 0.4,
    min_neighbors: int = 5,
) -> np.ndarray:
    """Radius Outlier Removal.

    A point is kept if it has at least `min_neighbors` neighbors within `radius`.

    Args:
        pts: (N, C) point cloud.
        radius: Search radius in meters.
        min_neighbors: Minimum neighbor count to be considered inlier.

    Returns:
        Boolean mask (N,) — True for inliers.
    """
    xyz = pts[:, :3].astype(np.float64)
    tree = cKDTree(xyz)

    # Count neighbors within radius (excluding self)
    counts = np.array(tree.query_ball_point(xyz, radius, workers=-1, return_length=True))
    counts = counts - 1  # exclude self

    inlier_mask = counts >= min_neighbors
    return inlier_mask


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_filter_on_dir(
    method: str,
    input_dir: str,
    out_dir: str,
    k: int = 20,
    alpha: float = 1.0,
    radius: float = 0.4,
    min_neighbors: int = 5,
    num_features: int = 4,
) -> None:
    """Apply SOR or ROR to every .bin file in input_dir, write to out_dir."""
    in_path = Path(input_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(in_path.glob("*.bin"))
    if not bin_files:
        logger.error(f"No .bin files found in {in_path}")
        return

    total_removed = 0
    total_points = 0
    t0 = time.time()

    for idx, bf in enumerate(bin_files):
        pts = np.fromfile(str(bf), dtype=np.float32).reshape(-1, num_features)
        total_points += len(pts)

        if method == "sor":
            mask = sor_filter(pts, k=k, alpha=alpha)
        elif method == "ror":
            mask = ror_filter(pts, radius=radius, min_neighbors=min_neighbors)
        else:
            raise ValueError(f"Unknown method: {method}")

        filtered = pts[mask]
        total_removed += len(pts) - len(filtered)

        filtered.astype(np.float32).tofile(str(out_path / bf.name))

        if (idx + 1) % 500 == 0 or idx == len(bin_files) - 1:
            elapsed = time.time() - t0
            logger.info(
                f"[{method}] {idx+1}/{len(bin_files)} frames, "
                f"removed {total_removed}/{total_points} pts "
                f"({100*total_removed/max(total_points,1):.1f}%), "
                f"{elapsed:.0f}s elapsed"
            )

    logger.info(f"{method} done: {len(bin_files)} frames -> {out_path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="E3.1/E3.2 Statistical Filters")
    parser.add_argument("--method", choices=["sor", "ror"], required=True)
    parser.add_argument("--input_dir", required=True, help="Dir with attacked .bin files")
    parser.add_argument("--out_dir", required=True, help="Output dir for defended .bin files")
    parser.add_argument("--k_neighbors", type=int, default=20, help="k for SOR kNN")
    parser.add_argument("--alpha", type=float, default=1.0, help="SOR threshold multiplier")
    parser.add_argument("--radius", type=float, default=0.4, help="ROR search radius (m)")
    parser.add_argument("--min_neighbors", type=int, default=5, help="ROR min neighbor count")
    parser.add_argument(
        "--num_features",
        type=int,
        default=4,
        help="Point feature dimension in each .bin file: KITTI=4, nuScenes=5.",
    )
    args = parser.parse_args()

    run_filter_on_dir(
        method=args.method,
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        k=args.k_neighbors,
        alpha=args.alpha,
        radius=args.radius,
        min_neighbors=args.min_neighbors,
        num_features=args.num_features,
    )


if __name__ == "__main__":
    main()
