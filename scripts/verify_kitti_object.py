"""Verify the KITTI 3D Object Detection directory layout used by ASAP.

This script checks that the training and testing splits each contain the
expected number of files for every modality. ImageSets are checked by a
separate script: scripts/verify_kitti_imagesets.py.

Usage:
    python scripts/verify_kitti_object.py
    python scripts/verify_kitti_object.py --root /root/autodl-tmp/ASAP/data/kitti
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPECTED = [
    ("training/velodyne", "*.bin", 7481),
    ("training/calib", "*.txt", 7481),
    ("training/label_2", "*.txt", 7481),
    ("training/image_2", "*.png", 7481),
    ("testing/velodyne", "*.bin", 7518),
    ("testing/calib", "*.txt", 7518),
    ("testing/image_2", "*.png", 7518),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/root/autodl-tmp/ASAP/data/kitti",
        help="KITTI 3D object detection root directory.",
    )
    args = parser.parse_args()
    root = Path(args.root)

    print(f"[ASAP] Verifying KITTI object data at: {root}")
    if not root.is_dir():
        print(f"[ASAP][FAIL] Directory not found: {root}")
        return 1

    failed = 0
    for rel, pattern, expected in EXPECTED:
        sub = root / rel
        if not sub.is_dir():
            print(f"[ASAP][FAIL] Missing directory: {sub}")
            failed += 1
            continue
        count = sum(1 for _ in sub.glob(pattern))
        status = "OK" if count == expected else "FAIL"
        if count != expected:
            failed += 1
        print(f"[ASAP][{status}] {rel}: {count} files (expected {expected})")

    if failed:
        print(f"[ASAP] KITTI object verification FAILED with {failed} issue(s).")
        return 1
    print("[ASAP] KITTI object verification PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
