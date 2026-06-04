"""E2.2 — Point-perturbation attack.

Perturb existing LiDAR points that belong to ground-truth objects by
adding bounded random displacements, simulating adversarial sensor noise.

Protocol follows Xiang et al. (CVPR 2019) — gradient-free geometric variant:
- For each target object, identify the LiDAR points inside the GT box.
- Displace each point by a random offset bounded by epsilon in L-inf norm.
- The perturbation degrades 3D shape evidence without inserting or
  removing any points.

Usage:
    python -m asap.attacks.point_perturbation \\
        --info_pkl data/kitti/kitti_infos_val.pkl \\
        --velodyne_dir data/kitti/training/velodyne \\
        --out_dir outputs/attacks/kitti_pointpillars/E2.2_perturbation \\
        --epsilon 0.3 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from asap.attacks.utils import (
    load_kitti_infos,
    load_velodyne,
    save_velodyne,
    points_in_box,
    write_attack_meta,
)

logger = logging.getLogger(__name__)

TARGET_CLASSES = {"Car", "Pedestrian", "Cyclist"}


def perturb_points_single_frame(
    pts: np.ndarray,
    gt_boxes: np.ndarray,
    gt_names: np.ndarray,
    epsilon: float = 0.3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Perturb object points by bounded random displacement.

    Args:
        pts: (N, 4) original point cloud [x, y, z, reflectance].
        gt_boxes: (M, 7) ground-truth boxes (non-DontCare).
        gt_names: (M_all,) class names for ALL objects.
        epsilon: Maximum per-axis displacement in meters (L-inf bound).
        rng: NumPy random generator.

    Returns:
        attacked_pts: (N, 4) with object points perturbed.
        meta: dict with perturbation statistics.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    non_dc_mask = gt_names != "DontCare"
    target_names = gt_names[non_dc_mask]

    attacked_pts = pts.copy()
    total_perturbed = 0
    n_targeted = 0

    for i, (name, box) in enumerate(zip(target_names, gt_boxes)):
        if name not in TARGET_CLASSES:
            continue
        n_targeted += 1

        # Find points inside this box
        in_mask = points_in_box(pts[:, :3], box)
        n_in = in_mask.sum()
        if n_in == 0:
            continue

        # Generate bounded random perturbation (L-inf)
        noise = rng.uniform(-epsilon, epsilon, size=(n_in, 3)).astype(np.float32)
        attacked_pts[in_mask, :3] += noise
        total_perturbed += n_in

    meta = {
        "n_targeted_objects": n_targeted,
        "n_perturbed_points": int(total_perturbed),
        "epsilon": epsilon,
    }
    return attacked_pts, meta


def run_perturbation_attack(
    info_pkl: str,
    velodyne_dir: str,
    out_dir: str,
    epsilon: float = 0.3,
    seed: int = 42,
    max_frames: int | None = None,
) -> None:
    """Run point-perturbation attack on a full val split."""
    infos = load_kitti_infos(info_pkl)
    out_path = Path(out_dir)
    vel_out = out_path / "velodyne_attacked"
    vel_out.mkdir(parents=True, exist_ok=True)
    meta_path = out_path / "attack_meta.jsonl"

    if meta_path.exists():
        meta_path.unlink()

    rng = np.random.default_rng(seed)
    n_frames = min(len(infos), max_frames) if max_frames else len(infos)
    total_perturbed = 0

    for idx in range(n_frames):
        info = infos[idx]
        frame_id = info["point_cloud"]["lidar_idx"]
        vel_file = Path(velodyne_dir) / f"{frame_id}.bin"

        pts = load_velodyne(vel_file)
        gt_boxes = info["annos"]["gt_boxes_lidar"]
        gt_names = info["annos"]["name"]

        attacked_pts, meta = perturb_points_single_frame(
            pts, gt_boxes, gt_names, epsilon=epsilon, rng=rng,
        )

        save_velodyne(attacked_pts, vel_out / f"{frame_id}.bin")
        write_attack_meta(meta_path, frame_id, "perturbation", meta)
        total_perturbed += meta["n_perturbed_points"]

        if (idx + 1) % 500 == 0 or idx == n_frames - 1:
            logger.info(
                f"[perturbation] {idx+1}/{n_frames} frames done, "
                f"total perturbed points so far: {total_perturbed}"
            )

    logger.info(
        f"Point-perturbation attack complete: {n_frames} frames, "
        f"{total_perturbed} total perturbed points -> {vel_out}"
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="E2.2 Point-perturbation attack")
    parser.add_argument("--info_pkl", required=True)
    parser.add_argument("--velodyne_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epsilon", type=float, default=0.3, help="L-inf bound (m)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    run_perturbation_attack(
        info_pkl=args.info_pkl,
        velodyne_dir=args.velodyne_dir,
        out_dir=args.out_dir,
        epsilon=args.epsilon,
        seed=args.seed,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
