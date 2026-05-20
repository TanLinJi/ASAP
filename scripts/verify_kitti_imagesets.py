"""Verify KITTI ImageSets split files used by OpenPCDet.

Checks that each split file exists, has the expected number of entries, and
that train and val do not overlap. trainval is checked to equal the union
of train and val.

Usage:
    python scripts/verify_kitti_imagesets.py
    python scripts/verify_kitti_imagesets.py --root /root/autodl-tmp/ASAP/data/kitti
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPECTED = {
    "train.txt": 3712,
    "val.txt": 3769,
    "trainval.txt": 7481,
    "test.txt": 7518,
}


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/root/autodl-tmp/ASAP/data/kitti",
        help="KITTI 3D object detection root directory.",
    )
    args = parser.parse_args()
    image_sets = Path(args.root) / "ImageSets"

    print(f"[ASAP] Verifying KITTI ImageSets at: {image_sets}")
    if not image_sets.is_dir():
        print(f"[ASAP][FAIL] Missing directory: {image_sets}")
        return 1

    failed = 0
    counts: dict[str, list[str]] = {}
    for name, expected in EXPECTED.items():
        path = image_sets / name
        if not path.is_file():
            print(f"[ASAP][FAIL] Missing file: {path}")
            failed += 1
            continue
        ids = read_ids(path)
        counts[name] = ids
        status = "OK" if len(ids) == expected else "FAIL"
        if len(ids) != expected:
            failed += 1
        print(f"[ASAP][{status}] {name}: {len(ids)} entries (expected {expected})")

    if "train.txt" in counts and "val.txt" in counts:
        overlap = set(counts["train.txt"]) & set(counts["val.txt"])
        status = "OK" if not overlap else "FAIL"
        if overlap:
            failed += 1
        print(f"[ASAP][{status}] train/val overlap: {len(overlap)} (expected 0)")

    if all(k in counts for k in ("train.txt", "val.txt", "trainval.txt")):
        union = set(counts["train.txt"]) | set(counts["val.txt"])
        trainval_set = set(counts["trainval.txt"])
        status = "OK" if union == trainval_set else "FAIL"
        if union != trainval_set:
            failed += 1
        print(
            f"[ASAP][{status}] trainval == train ∪ val "
            f"(|trainval|={len(trainval_set)}, |train ∪ val|={len(union)})"
        )

    if failed:
        print(f"[ASAP] ImageSets verification FAILED with {failed} issue(s).")
        return 1
    print("[ASAP] ImageSets verification PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
