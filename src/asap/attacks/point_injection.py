"""E2.1 — Point-injection attack.

Insert adversarial point clusters near ground-truth object surfaces to
create ghost detections or shift existing bounding boxes.

Protocol follows Cao et al. (CCS 2019) / Tu et al. (CVPR 2020):
- For each target object in the GT annotation, insert K_inj points
  sampled uniformly inside a thin shell around the box surface.
- The injected cluster mimics an additional small object or surface
  extension that the detector must now reconcile.
- This is a *geometric* (gradient-free) injection: it does not require
  victim model gradients and is therefore detector-agnostic.

Usage:
    python -m asap.attacks.point_injection \\
        --info_pkl data/kitti/kitti_infos_val.pkl \\
        --velodyne_dir data/kitti/training/velodyne \\
        --out_dir outputs/attacks/kitti_pointpillars/E2.1_injection \\
        --k_inj 50 --r_inj 0.5 --seed 42
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
    sample_points_in_box,
    write_attack_meta,
)

logger = logging.getLogger(__name__)

TARGET_CLASSES = {"Car", "Pedestrian", "Cyclist"}


def inject_points_single_frame(
    pts: np.ndarray,
    gt_boxes: np.ndarray,
    gt_names: np.ndarray,
    k_inj: int = 50,
    r_inj: float = 0.5,
    reflectance: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Inject adversarial point clusters near each target GT box.

    Args:
        pts: (N, 4) original point cloud [x, y, z, reflectance].
        gt_boxes: (M, 7) ground-truth boxes [x, y, z, dx, dy, dz, heading].
        gt_names: (M_all,) class names for ALL objects (including DontCare).
        k_inj: Number of points to inject per target object.
        r_inj: Expansion radius — injected points are placed inside a
               box that is r_inj wider in each dimension than the GT box.
        reflectance: Reflectance value assigned to injected points.
        rng: NumPy random generator.

    Returns:
        attacked_pts: (N + n_injected, 4) attacked point cloud.
        meta: dict with injection statistics.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Only attack target classes; gt_boxes already excludes DontCare in
    # the OpenPCDet info format, so we filter gt_names by non-DontCare.
    non_dc_mask = gt_names != "DontCare"
    target_names = gt_names[non_dc_mask]

    # gt_boxes aligns with non-DontCare entries
    assert len(gt_boxes) == non_dc_mask.sum(), (
        f"gt_boxes ({len(gt_boxes)}) vs non-DontCare ({non_dc_mask.sum()}) mismatch"
    )

    injected_all = []
    n_targeted = 0

    for i, (name, box) in enumerate(zip(target_names, gt_boxes)):
        if name not in TARGET_CLASSES:
            continue
        n_targeted += 1

        # Create an expanded box for injection placement
        expanded_box = box.copy()
        expanded_box[3] += 2 * r_inj  # dx
        expanded_box[4] += 2 * r_inj  # dy
        expanded_box[5] += 2 * r_inj  # dz

        # Sample points inside expanded box
        inj_xyz = sample_points_in_box(expanded_box, k_inj, rng)

        # Add reflectance column
        inj_ref = np.full((k_inj, 1), reflectance, dtype=np.float32)
        inj_pts = np.hstack([inj_xyz, inj_ref])
        injected_all.append(inj_pts)

    if injected_all:
        injected = np.vstack(injected_all)
        attacked_pts = np.vstack([pts, injected])
    else:
        injected = np.zeros((0, 4), dtype=np.float32)
        attacked_pts = pts.copy()

    meta = {
        "n_targeted_objects": n_targeted,
        "n_injected_points": len(injected),
        "k_inj": k_inj,
        "r_inj": r_inj,
    }
    return attacked_pts, meta


def run_injection_attack(
    info_pkl: str,
    velodyne_dir: str,
    out_dir: str,
    k_inj: int = 50,
    r_inj: float = 0.5,
    seed: int = 42,
    max_frames: int | None = None,
) -> None:
    """Run point-injection attack on a full val split."""
    infos = load_kitti_infos(info_pkl)
    out_path = Path(out_dir)
    vel_out = out_path / "velodyne_attacked"
    vel_out.mkdir(parents=True, exist_ok=True)
    meta_path = out_path / "attack_meta.jsonl"

    # Clear previous meta
    if meta_path.exists():
        meta_path.unlink()

    rng = np.random.default_rng(seed)

    n_frames = min(len(infos), max_frames) if max_frames else len(infos)
    total_injected = 0

    for idx in range(n_frames):
        info = infos[idx]
        frame_id = info["point_cloud"]["lidar_idx"]
        vel_file = Path(velodyne_dir) / f"{frame_id}.bin"

        pts = load_velodyne(vel_file)
        gt_boxes = info["annos"]["gt_boxes_lidar"]
        gt_names = info["annos"]["name"]

        attacked_pts, meta = inject_points_single_frame(
            pts, gt_boxes, gt_names,
            k_inj=k_inj, r_inj=r_inj, rng=rng,
        )

        save_velodyne(attacked_pts, vel_out / f"{frame_id}.bin")
        write_attack_meta(meta_path, frame_id, "injection", meta)
        total_injected += meta["n_injected_points"]

        if (idx + 1) % 500 == 0 or idx == n_frames - 1:
            logger.info(
                f"[injection] {idx+1}/{n_frames} frames done, "
                f"total injected points so far: {total_injected}"
            )

    logger.info(
        f"Point-injection attack complete: {n_frames} frames, "
        f"{total_injected} total injected points -> {vel_out}"
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="E2.1 Point-injection attack")
    parser.add_argument("--info_pkl", required=True, help="Path to kitti_infos_val.pkl")
    parser.add_argument("--velodyne_dir", required=True, help="Path to clean velodyne/ dir")
    parser.add_argument("--out_dir", required=True, help="Output directory for attacked bins")
    parser.add_argument("--k_inj", type=int, default=50, help="Injected points per object")
    parser.add_argument("--r_inj", type=float, default=0.5, help="Expansion radius (m)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames for dev")
    args = parser.parse_args()

    run_injection_attack(
        info_pkl=args.info_pkl,
        velodyne_dir=args.velodyne_dir,
        out_dir=args.out_dir,
        k_inj=args.k_inj,
        r_inj=args.r_inj,
        seed=args.seed,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
