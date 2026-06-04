"""E2.3 — Point-dropping attack.

Remove a fraction of LiDAR points that belong to ground-truth objects,
simulating adversarial sensor spoofing or physical occlusion.

Protocol follows Zheng et al. (ICCV 2019) — saliency-agnostic variant:
- For each target object, identify the LiDAR points inside the GT box.
- Randomly drop a fraction `drop_ratio` of those points.
- Background points are untouched.

Usage:
    python -m asap.attacks.point_dropping \\
        --info_pkl data/kitti/kitti_infos_val.pkl \\
        --velodyne_dir data/kitti/training/velodyne \\
        --out_dir outputs/attacks/kitti_pointpillars/E2.3_dropping \\
        --drop_ratio 0.5 --seed 42
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


def drop_points_single_frame(
    pts: np.ndarray,
    gt_boxes: np.ndarray,
    gt_names: np.ndarray,
    drop_ratio: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Drop a fraction of object points.

    Args:
        pts: (N, 4) original point cloud [x, y, z, reflectance].
        gt_boxes: (M, 7) ground-truth boxes (non-DontCare).
        gt_names: (M_all,) class names for ALL objects.
        drop_ratio: Fraction of object points to remove (0-1).
        rng: NumPy random generator.

    Returns:
        attacked_pts: Remaining points after dropping.
        meta: dict with dropping statistics.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    non_dc_mask = gt_names != "DontCare"
    target_names = gt_names[non_dc_mask]

    # Collect all indices to drop across all target objects
    drop_indices = set()
    n_targeted = 0
    n_total_in_objects = 0

    for i, (name, box) in enumerate(zip(target_names, gt_boxes)):
        if name not in TARGET_CLASSES:
            continue
        n_targeted += 1

        in_mask = points_in_box(pts[:, :3], box)
        obj_indices = np.where(in_mask)[0]
        n_in = len(obj_indices)
        n_total_in_objects += n_in

        if n_in == 0:
            continue

        # Randomly select points to drop
        n_drop = max(1, int(n_in * drop_ratio))
        chosen = rng.choice(obj_indices, size=n_drop, replace=False)
        drop_indices.update(chosen.tolist())

    # Create mask of points to keep
    keep_mask = np.ones(len(pts), dtype=bool)
    if drop_indices:
        keep_mask[list(drop_indices)] = False

    attacked_pts = pts[keep_mask]

    meta = {
        "n_targeted_objects": n_targeted,
        "n_original_points": int(len(pts)),
        "n_points_in_objects": int(n_total_in_objects),
        "n_dropped_points": int(len(drop_indices)),
        "n_remaining_points": int(len(attacked_pts)),
        "drop_ratio": drop_ratio,
    }
    return attacked_pts, meta


def run_dropping_attack(
    info_pkl: str,
    velodyne_dir: str,
    out_dir: str,
    drop_ratio: float = 0.5,
    seed: int = 42,
    max_frames: int | None = None,
) -> None:
    """Run point-dropping attack on a full val split."""
    infos = load_kitti_infos(info_pkl)
    out_path = Path(out_dir)
    vel_out = out_path / "velodyne_attacked"
    vel_out.mkdir(parents=True, exist_ok=True)
    meta_path = out_path / "attack_meta.jsonl"

    if meta_path.exists():
        meta_path.unlink()

    rng = np.random.default_rng(seed)
    n_frames = min(len(infos), max_frames) if max_frames else len(infos)
    total_dropped = 0

    for idx in range(n_frames):
        info = infos[idx]
        frame_id = info["point_cloud"]["lidar_idx"]
        vel_file = Path(velodyne_dir) / f"{frame_id}.bin"

        pts = load_velodyne(vel_file)
        gt_boxes = info["annos"]["gt_boxes_lidar"]
        gt_names = info["annos"]["name"]

        attacked_pts, meta = drop_points_single_frame(
            pts, gt_boxes, gt_names, drop_ratio=drop_ratio, rng=rng,
        )

        save_velodyne(attacked_pts, vel_out / f"{frame_id}.bin")
        write_attack_meta(meta_path, frame_id, "dropping", meta)
        total_dropped += meta["n_dropped_points"]

        if (idx + 1) % 500 == 0 or idx == n_frames - 1:
            logger.info(
                f"[dropping] {idx+1}/{n_frames} frames done, "
                f"total dropped points so far: {total_dropped}"
            )

    logger.info(
        f"Point-dropping attack complete: {n_frames} frames, "
        f"{total_dropped} total dropped points -> {vel_out}"
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="E2.3 Point-dropping attack")
    parser.add_argument("--info_pkl", required=True)
    parser.add_argument("--velodyne_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--drop_ratio", type=float, default=0.5, help="Fraction to drop")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    run_dropping_attack(
        info_pkl=args.info_pkl,
        velodyne_dir=args.velodyne_dir,
        out_dir=args.out_dir,
        drop_ratio=args.drop_ratio,
        seed=args.seed,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
