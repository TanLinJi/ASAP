#!/usr/bin/env bash
# Download the four KITTI 3D Object Detection zip archives required by ASAP.
#
# Output:
#   ${DOWNLOAD_DIR}/data_object_velodyne.zip
#   ${DOWNLOAD_DIR}/data_object_calib.zip
#   ${DOWNLOAD_DIR}/data_object_label_2.zip
#   ${DOWNLOAD_DIR}/data_object_image_2.zip
#   ${KITTI_ROOT}/training/{velodyne,calib,label_2,image_2}/
#   ${KITTI_ROOT}/testing/{velodyne,calib,image_2}/
#
# Environment variables (with defaults):
#   ASAP_ROOT      default: /root/autodl-tmp/ASAP
#   KITTI_ROOT     default: ${ASAP_ROOT}/data/kitti
#   DOWNLOAD_DIR   default: ${ASAP_ROOT}/data/downloads
#
# Usage:
#   bash scripts/download_kitti_object.sh

set -euo pipefail

ASAP_ROOT="${ASAP_ROOT:-/root/autodl-tmp/ASAP}"
KITTI_ROOT="${KITTI_ROOT:-${ASAP_ROOT}/data/kitti}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${ASAP_ROOT}/data/downloads}"
KITTI_S3_BASE="https://s3.eu-central-1.amazonaws.com/avg-kitti"

mkdir -p "${KITTI_ROOT}" "${DOWNLOAD_DIR}"

echo "[ASAP] Downloading KITTI 3D Object Detection archives to ${DOWNLOAD_DIR}"
wget -c -P "${DOWNLOAD_DIR}" "${KITTI_S3_BASE}/data_object_velodyne.zip"
wget -c -P "${DOWNLOAD_DIR}" "${KITTI_S3_BASE}/data_object_calib.zip"
wget -c -P "${DOWNLOAD_DIR}" "${KITTI_S3_BASE}/data_object_label_2.zip"
wget -c -P "${DOWNLOAD_DIR}" "${KITTI_S3_BASE}/data_object_image_2.zip"

echo "[ASAP] Extracting archives to ${KITTI_ROOT}"
unzip -n "${DOWNLOAD_DIR}/data_object_velodyne.zip" -d "${KITTI_ROOT}"
unzip -n "${DOWNLOAD_DIR}/data_object_calib.zip" -d "${KITTI_ROOT}"
unzip -n "${DOWNLOAD_DIR}/data_object_label_2.zip" -d "${KITTI_ROOT}"
unzip -n "${DOWNLOAD_DIR}/data_object_image_2.zip" -d "${KITTI_ROOT}"

echo "[ASAP] KITTI object data ready under ${KITTI_ROOT}"
