"""nuScenes keyframe geometric attacks for E2.x@nuScenes.

This module attacks only the current keyframe file referenced by each
OpenPCDet nuScenes info record (`info["lidar_path"]`). The detector may still
load historical sweeps according to its config. This is intentional for the
first nuScenes loop and must be reported as a keyframe-only attack protocol.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

from asap.attacks.utils import points_in_box, sample_points_in_box, write_attack_meta

logger = logging.getLogger(__name__)

NUSCENES_TARGET_CLASSES = {
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
}


def load_nuscenes_points(path: str | Path) -> np.ndarray:
    """Load a nuScenes `.pcd.bin` file as (N, 5) float32."""
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 5)


def save_nuscenes_points(points: np.ndarray, path: str | Path) -> None:
    """Save a nuScenes `.pcd.bin` file, preserving five float32 columns."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    points.astype(np.float32).tofile(str(path))


def load_nuscenes_infos(path: str | Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _parse_target_classes(value: str) -> set[str]:
    if value == "all":
        return set(NUSCENES_TARGET_CLASSES)
    return {item.strip() for item in value.split(",") if item.strip()}


def _target_iter(
    gt_boxes: np.ndarray,
    gt_names: np.ndarray,
    target_classes: set[str],
):
    for name, box in zip(gt_names, gt_boxes[:, :7]):
        name = str(name)
        if name in target_classes:
            yield name, box


def inject_keyframe(
    points: np.ndarray,
    gt_boxes: np.ndarray,
    gt_names: np.ndarray,
    target_classes: set[str],
    k_inj: int,
    r_inj: float,
    reflectance: float,
    extra_value: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    injected_all = []
    n_targeted = 0

    for _name, box in _target_iter(gt_boxes, gt_names, target_classes):
        n_targeted += 1
        expanded = box.copy()
        expanded[3:6] += 2.0 * r_inj
        xyz = sample_points_in_box(expanded, k_inj, rng)
        tail = np.full((k_inj, 2), [reflectance, extra_value], dtype=np.float32)
        injected_all.append(np.hstack([xyz, tail]))

    if injected_all:
        injected = np.vstack(injected_all).astype(np.float32)
        out = np.vstack([points, injected]).astype(np.float32)
    else:
        injected = np.zeros((0, 5), dtype=np.float32)
        out = points.copy()

    return out, {
        "n_targeted_objects": n_targeted,
        "n_injected_points": int(len(injected)),
        "k_inj": int(k_inj),
        "r_inj": float(r_inj),
    }


def perturb_keyframe(
    points: np.ndarray,
    gt_boxes: np.ndarray,
    gt_names: np.ndarray,
    target_classes: set[str],
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    out = points.copy()
    n_targeted = 0
    n_perturbed = 0

    for _name, box in _target_iter(gt_boxes, gt_names, target_classes):
        n_targeted += 1
        mask = points_in_box(points[:, :3], box)
        n_in = int(mask.sum())
        if n_in == 0:
            continue
        noise = rng.uniform(-epsilon, epsilon, size=(n_in, 3)).astype(np.float32)
        out[mask, :3] += noise
        n_perturbed += n_in

    return out, {
        "n_targeted_objects": n_targeted,
        "n_perturbed_points": int(n_perturbed),
        "epsilon": float(epsilon),
    }


def drop_keyframe(
    points: np.ndarray,
    gt_boxes: np.ndarray,
    gt_names: np.ndarray,
    target_classes: set[str],
    drop_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    drop_indices: set[int] = set()
    n_targeted = 0
    n_points_in_objects = 0

    for _name, box in _target_iter(gt_boxes, gt_names, target_classes):
        n_targeted += 1
        mask = points_in_box(points[:, :3], box)
        obj_indices = np.where(mask)[0]
        n_in = int(len(obj_indices))
        n_points_in_objects += n_in
        if n_in == 0:
            continue
        n_drop = max(1, int(n_in * drop_ratio))
        chosen = rng.choice(obj_indices, size=n_drop, replace=False)
        drop_indices.update(int(x) for x in chosen)

    keep = np.ones(len(points), dtype=bool)
    if drop_indices:
        keep[list(drop_indices)] = False
    out = points[keep]

    return out, {
        "n_targeted_objects": n_targeted,
        "n_original_points": int(len(points)),
        "n_points_in_objects": int(n_points_in_objects),
        "n_dropped_points": int(len(drop_indices)),
        "n_remaining_points": int(len(out)),
        "drop_ratio": float(drop_ratio),
    }


def run_attack(
    info_pkl: str,
    nuscenes_root: str,
    out_dir: str,
    attack: str,
    target_classes: str = "all",
    k_inj: int = 50,
    r_inj: float = 0.5,
    reflectance: float = 0.5,
    extra_value: float = 0.0,
    epsilon: float = 0.3,
    drop_ratio: float = 0.5,
    seed: int = 42,
    max_frames: int | None = None,
) -> None:
    infos = load_nuscenes_infos(info_pkl)
    root = Path(nuscenes_root)
    out_root = Path(out_dir)
    lidar_out = out_root / "samples" / "LIDAR_TOP"
    lidar_out.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "attack_meta.jsonl"
    if meta_path.exists():
        meta_path.unlink()

    targets = _parse_target_classes(target_classes)
    rng = np.random.default_rng(seed)
    n_frames = min(len(infos), max_frames) if max_frames else len(infos)

    totals = {
        "injected": 0,
        "perturbed": 0,
        "dropped": 0,
        "targeted": 0,
    }

    for idx, info in enumerate(infos[:n_frames]):
        lidar_rel = Path(info["lidar_path"])
        frame_id = lidar_rel.stem
        points = load_nuscenes_points(root / lidar_rel)
        gt_boxes = info["gt_boxes"]
        gt_names = info["gt_names"]

        if attack == "injection":
            attacked, meta = inject_keyframe(
                points,
                gt_boxes,
                gt_names,
                targets,
                k_inj=k_inj,
                r_inj=r_inj,
                reflectance=reflectance,
                extra_value=extra_value,
                rng=rng,
            )
            totals["injected"] += meta["n_injected_points"]
        elif attack == "perturbation":
            attacked, meta = perturb_keyframe(
                points,
                gt_boxes,
                gt_names,
                targets,
                epsilon=epsilon,
                rng=rng,
            )
            totals["perturbed"] += meta["n_perturbed_points"]
        elif attack == "dropping":
            attacked, meta = drop_keyframe(
                points,
                gt_boxes,
                gt_names,
                targets,
                drop_ratio=drop_ratio,
                rng=rng,
            )
            totals["dropped"] += meta["n_dropped_points"]
        else:
            raise ValueError(f"Unknown attack: {attack}")

        totals["targeted"] += meta["n_targeted_objects"]
        save_nuscenes_points(attacked, lidar_out / lidar_rel.name)
        write_attack_meta(
            meta_path,
            frame_id,
            f"nuscenes_keyframe_{attack}",
            {
                "source_lidar_path": str(lidar_rel),
                "target_classes": sorted(targets),
                **meta,
            },
        )

        if (idx + 1) % 500 == 0 or idx == n_frames - 1:
            logger.info(
                "[nuscenes-%s] %d/%d frames, targeted=%d, "
                "injected=%d, perturbed=%d, dropped=%d",
                attack,
                idx + 1,
                n_frames,
                totals["targeted"],
                totals["injected"],
                totals["perturbed"],
                totals["dropped"],
            )

    logger.info("nuScenes keyframe %s attack complete -> %s", attack, lidar_out)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info_pkl", required=True)
    parser.add_argument("--nuscenes_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--attack",
        choices=["injection", "perturbation", "dropping"],
        required=True,
    )
    parser.add_argument("--target_classes", default="all")
    parser.add_argument("--k_inj", type=int, default=50)
    parser.add_argument("--r_inj", type=float, default=0.5)
    parser.add_argument("--reflectance", type=float, default=0.5)
    parser.add_argument("--extra_value", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--drop_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    run_attack(**vars(args))


if __name__ == "__main__":
    main()
