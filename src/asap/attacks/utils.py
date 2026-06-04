"""Shared utilities for ASAP adversarial attacks on KITTI / nuScenes."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_velodyne(path: str | Path) -> np.ndarray:
    """Load a KITTI-format velodyne bin (N, 4) float32: x y z reflectance."""
    pts = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
    return pts


def save_velodyne(pts: np.ndarray, path: str | Path) -> None:
    """Save points as a KITTI-format .bin."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pts.astype(np.float32).tofile(str(path))


def load_kitti_infos(pkl_path: str | Path) -> List[dict]:
    """Load an OpenPCDet-style kitti_infos_*.pkl."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def boxes_to_corners_3d(boxes: np.ndarray) -> np.ndarray:
    """Convert (N, 7) boxes [x,y,z,dx,dy,dz,heading] to (N, 8, 3) corners.

    Uses the OpenPCDet convention:
    - x forward, y left, z up.
    - dx = length along x, dy = width along y, dz = height along z.
    - heading rotates around z-axis (counter-clockwise from x-axis).
    - z is at the *center* of the box.
    """
    N = boxes.shape[0]
    corners = np.zeros((N, 8, 3), dtype=np.float32)

    dx, dy, dz = boxes[:, 3] / 2, boxes[:, 4] / 2, boxes[:, 5] / 2
    heading = boxes[:, 6]

    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    # 8 corners relative to center, before rotation
    #   order: [±dx, ±dy, ±dz]
    x_corners = np.stack([dx, dx, -dx, -dx, dx, dx, -dx, -dx], axis=1)   # (N,8)
    y_corners = np.stack([dy, -dy, -dy, dy, dy, -dy, -dy, dy], axis=1)
    z_corners = np.stack([dz, dz, dz, dz, -dz, -dz, -dz, -dz], axis=1)

    # Rotate around z
    corners[:, :, 0] = cos_h[:, None] * x_corners - sin_h[:, None] * y_corners + boxes[:, 0:1]
    corners[:, :, 1] = sin_h[:, None] * x_corners + cos_h[:, None] * y_corners + boxes[:, 1:2]
    corners[:, :, 2] = z_corners + boxes[:, 2:3]

    return corners


def points_in_box(pts_xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Return boolean mask of pts_xyz (N,3) inside a single 7-DOF box.

    box: [x, y, z, dx, dy, dz, heading].
    """
    cx, cy, cz, dx, dy, dz, heading = box
    cos_h, sin_h = np.cos(-heading), np.sin(-heading)

    # Translate to box center
    local_x = pts_xyz[:, 0] - cx
    local_y = pts_xyz[:, 1] - cy
    local_z = pts_xyz[:, 2] - cz

    # Rotate to box-aligned frame
    rot_x = cos_h * local_x - sin_h * local_y
    rot_y = sin_h * local_x + cos_h * local_y

    mask = (
        (np.abs(rot_x) <= dx / 2)
        & (np.abs(rot_y) <= dy / 2)
        & (np.abs(local_z) <= dz / 2)
    )
    return mask


def sample_points_in_box(
    box: np.ndarray, n_points: int, rng: np.random.Generator
) -> np.ndarray:
    """Uniformly sample `n_points` 3D locations inside a 7-DOF box.

    Returns (n_points, 3) array.
    """
    cx, cy, cz, dx, dy, dz, heading = box
    # Sample in box-aligned frame
    local = rng.uniform(
        low=[-dx / 2, -dy / 2, -dz / 2],
        high=[dx / 2, dy / 2, dz / 2],
        size=(n_points, 3),
    ).astype(np.float32)

    # Rotate back to world
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    world = np.empty_like(local)
    world[:, 0] = cos_h * local[:, 0] - sin_h * local[:, 1] + cx
    world[:, 1] = sin_h * local[:, 0] + cos_h * local[:, 1] + cy
    world[:, 2] = local[:, 2] + cz
    return world


def write_attack_meta(
    path: str | Path, frame_id: str, attack_type: str, meta: dict
) -> None:
    """Append a JSONL line to the attack metadata log."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    entry = {"frame_id": frame_id, "attack_type": attack_type, **meta}
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=_json_default) + "\n")


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
