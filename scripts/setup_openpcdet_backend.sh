#!/usr/bin/env bash
# Prepare an OpenPCDet backend directory for ASAP without downloading data into OpenPCDet.
#
# This script assumes OpenPCDet has already been cloned manually to:
#   ${ASAP_ROOT}/third_party/OpenPCDet
#
# It checks the backend, verifies that the nested OpenPCDet .git directory has
# been removed, and creates this symlink:
#   ${OPENPCDET_ROOT}/data/kitti -> ${KITTI_ROOT}
#
# Environment variables (with defaults):
#   ASAP_ROOT         default: /root/autodl-tmp/ASAP
#   KITTI_ROOT        default: ${ASAP_ROOT}/data/kitti
#   OPENPCDET_ROOT    default: ${ASAP_ROOT}/third_party/OpenPCDet
#
# Usage after manually cloning OpenPCDet:
#   bash scripts/setup_openpcdet_backend.sh

set -euo pipefail

ASAP_ROOT="${ASAP_ROOT:-/root/autodl-tmp/ASAP}"
KITTI_ROOT="${KITTI_ROOT:-${ASAP_ROOT}/data/kitti}"
OPENPCDET_ROOT="${OPENPCDET_ROOT:-${ASAP_ROOT}/third_party/OpenPCDet}"
OPENPCDET_DATA_DIR="${OPENPCDET_ROOT}/data"
OPENPCDET_KITTI_LINK="${OPENPCDET_DATA_DIR}/kitti"

fail() {
  echo "[ASAP][FAIL] $*" >&2
  exit 1
}

info() {
  echo "[ASAP] $*"
}

info "ASAP_ROOT=${ASAP_ROOT}"
info "KITTI_ROOT=${KITTI_ROOT}"
info "OPENPCDET_ROOT=${OPENPCDET_ROOT}"

if [[ ! -d "${KITTI_ROOT}" ]]; then
  fail "KITTI root does not exist: ${KITTI_ROOT}. Run scripts/prepare_kitti.sh first."
fi

if [[ ! -d "${OPENPCDET_ROOT}" ]]; then
  fail "OpenPCDet is not found at ${OPENPCDET_ROOT}. Manually clone it first, then remove its inner .git directory after recording the upstream commit."
fi

if [[ -d "${OPENPCDET_ROOT}/.git" ]]; then
  fail "Nested Git repository found at ${OPENPCDET_ROOT}/.git. Record the upstream commit, then remove it with: rm -rf ${OPENPCDET_ROOT}/.git"
fi

[[ -f "${OPENPCDET_ROOT}/setup.py" ]] || fail "Missing OpenPCDet setup.py under ${OPENPCDET_ROOT}"
[[ -f "${OPENPCDET_ROOT}/tools/cfgs/dataset_configs/kitti_dataset.yaml" ]] || fail "Missing OpenPCDet KITTI dataset config"
[[ -f "${OPENPCDET_ROOT}/tools/cfgs/kitti_models/pointpillar.yaml" ]] || fail "Missing OpenPCDet PointPillars KITTI config"

mkdir -p "${OPENPCDET_DATA_DIR}"

if [[ -L "${OPENPCDET_KITTI_LINK}" ]]; then
  target="$(readlink -f "${OPENPCDET_KITTI_LINK}")"
  expected="$(readlink -f "${KITTI_ROOT}")"
  if [[ "${target}" != "${expected}" ]]; then
    fail "Existing symlink ${OPENPCDET_KITTI_LINK} points to ${target}, expected ${expected}"
  fi
  info "Existing data/kitti symlink is correct."
elif [[ -e "${OPENPCDET_KITTI_LINK}" ]]; then
  fail "${OPENPCDET_KITTI_LINK} exists but is not a symlink. Move it away before continuing."
else
  ln -s "${KITTI_ROOT}" "${OPENPCDET_KITTI_LINK}"
  info "Created symlink: ${OPENPCDET_KITTI_LINK} -> ${KITTI_ROOT}"
fi

python "${ASAP_ROOT}/scripts/verify_kitti_object.py" --root "${KITTI_ROOT}"
python "${ASAP_ROOT}/scripts/verify_kitti_imagesets.py" --root "${KITTI_ROOT}"

info "OpenPCDet backend is ready for KITTI info generation."
