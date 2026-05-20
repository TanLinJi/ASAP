#!/usr/bin/env bash
# Download KITTI ImageSets split files used by OpenPCDet.
#
# Output:
#   ${KITTI_ROOT}/ImageSets/train.txt        3712 entries
#   ${KITTI_ROOT}/ImageSets/val.txt          3769 entries
#   ${KITTI_ROOT}/ImageSets/test.txt         7518 entries
#   ${KITTI_ROOT}/ImageSets/trainval.txt     7481 entries  (built from train + val)
#
# Note:
#   The OpenPCDet repository does not provide trainval.txt directly, so this
#   script generates trainval.txt by concatenating train.txt and val.txt.
#
# Environment variables (with defaults):
#   ASAP_ROOT      default: /root/autodl-tmp/ASAP
#   KITTI_ROOT     default: ${ASAP_ROOT}/data/kitti
#
# Usage:
#   bash scripts/download_kitti_imagesets.sh

set -euo pipefail

ASAP_ROOT="${ASAP_ROOT:-/root/autodl-tmp/ASAP}"
KITTI_ROOT="${KITTI_ROOT:-${ASAP_ROOT}/data/kitti}"
IMAGESETS_DIR="${KITTI_ROOT}/ImageSets"
OPENPCDET_IMAGESETS_BASE="https://raw.githubusercontent.com/open-mmlab/OpenPCDet/master/data/kitti/ImageSets"

mkdir -p "${IMAGESETS_DIR}"

echo "[ASAP] Downloading KITTI ImageSets to ${IMAGESETS_DIR}"
wget -O "${IMAGESETS_DIR}/train.txt" "${OPENPCDET_IMAGESETS_BASE}/train.txt"
wget -O "${IMAGESETS_DIR}/val.txt"   "${OPENPCDET_IMAGESETS_BASE}/val.txt"
wget -O "${IMAGESETS_DIR}/test.txt"  "${OPENPCDET_IMAGESETS_BASE}/test.txt"

echo "[ASAP] Building trainval.txt from train.txt + val.txt"
KITTI_ROOT="${KITTI_ROOT}" python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['KITTI_ROOT']) / 'ImageSets'
train = (root / 'train.txt').read_text().splitlines()
val = (root / 'val.txt').read_text().splitlines()
(root / 'trainval.txt').write_text('\n'.join(train + val) + '\n')
print(f"trainval.txt: {len(train) + len(val)} entries")
PY

echo "[ASAP] KITTI ImageSets ready under ${IMAGESETS_DIR}"
