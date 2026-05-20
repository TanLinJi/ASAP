#!/usr/bin/env bash
# End-to-end KITTI preparation for ASAP:
#   1. Download and extract the four KITTI 3D Object Detection archives.
#   2. Download ImageSets and build trainval.txt.
#   3. Verify the KITTI object data and ImageSets.
#
# Environment variables (with defaults):
#   ASAP_ROOT      default: /root/autodl-tmp/ASAP
#   KITTI_ROOT     default: ${ASAP_ROOT}/data/kitti
#   DOWNLOAD_DIR   default: ${ASAP_ROOT}/data/downloads
#
# Usage:
#   bash scripts/prepare_kitti.sh

set -euo pipefail

ASAP_ROOT="${ASAP_ROOT:-/root/autodl-tmp/ASAP}"
KITTI_ROOT="${KITTI_ROOT:-${ASAP_ROOT}/data/kitti}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${ASAP_ROOT}/data/downloads}"

export ASAP_ROOT KITTI_ROOT DOWNLOAD_DIR

bash "${ASAP_ROOT}/scripts/download_kitti_object.sh"
bash "${ASAP_ROOT}/scripts/download_kitti_imagesets.sh"

python "${ASAP_ROOT}/scripts/verify_kitti_object.py" --root "${KITTI_ROOT}"
python "${ASAP_ROOT}/scripts/verify_kitti_imagesets.py" --root "${KITTI_ROOT}"
