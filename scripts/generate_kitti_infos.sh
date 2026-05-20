#!/usr/bin/env bash
# Generate OpenPCDet KITTI info files for ASAP.
#
# Prerequisites:
#   1. KITTI object data and ImageSets are prepared under ${KITTI_ROOT}.
#   2. OpenPCDet exists under ${OPENPCDET_ROOT}.
#   3. ${OPENPCDET_ROOT}/data/kitti points to ${KITTI_ROOT}.
#   4. OpenPCDet dependencies are installed in the active Python environment.
#
# Environment variables (with defaults):
#   ASAP_ROOT         default: /root/autodl-tmp/ASAP
#   KITTI_ROOT        default: ${ASAP_ROOT}/data/kitti
#   OPENPCDET_ROOT    default: ${ASAP_ROOT}/third_party/OpenPCDet
#   PYTHON            default: python
#
# Usage:
#   bash scripts/generate_kitti_infos.sh

set -euo pipefail

ASAP_ROOT="${ASAP_ROOT:-/root/autodl-tmp/ASAP}"
KITTI_ROOT="${KITTI_ROOT:-${ASAP_ROOT}/data/kitti}"
OPENPCDET_ROOT="${OPENPCDET_ROOT:-${ASAP_ROOT}/third_party/OpenPCDet}"
PYTHON_BIN="${PYTHON:-python}"
DATASET_CFG="tools/cfgs/dataset_configs/kitti_dataset.yaml"

fail() {
  echo "[ASAP][FAIL] $*" >&2
  exit 1
}

info() {
  echo "[ASAP] $*"
}

[[ -d "${OPENPCDET_ROOT}" ]] || fail "OpenPCDet root not found: ${OPENPCDET_ROOT}"
[[ -f "${OPENPCDET_ROOT}/${DATASET_CFG}" ]] || fail "Missing dataset config: ${OPENPCDET_ROOT}/${DATASET_CFG}"
[[ -e "${OPENPCDET_ROOT}/data/kitti" ]] || fail "Missing ${OPENPCDET_ROOT}/data/kitti. Run scripts/setup_openpcdet_backend.sh first."

link_target="$(readlink -f "${OPENPCDET_ROOT}/data/kitti")"
expected_target="$(readlink -f "${KITTI_ROOT}")"
if [[ "${link_target}" != "${expected_target}" ]]; then
  fail "OpenPCDet data/kitti points to ${link_target}, expected ${expected_target}"
fi

python "${ASAP_ROOT}/scripts/verify_kitti_object.py" --root "${KITTI_ROOT}"
python "${ASAP_ROOT}/scripts/verify_kitti_imagesets.py" --root "${KITTI_ROOT}"

TORCH_LIB_DIR="$("${PYTHON_BIN}" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
if [[ -d "${TORCH_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

info "Generating KITTI infos from ${OPENPCDET_ROOT}"
(
  cd "${OPENPCDET_ROOT}"
  "${PYTHON_BIN}" -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos "${DATASET_CFG}"
)

expected_outputs=(
  "kitti_infos_train.pkl"
  "kitti_infos_val.pkl"
  "kitti_infos_trainval.pkl"
  "kitti_dbinfos_train.pkl"
  "gt_database"
)

missing=0
for name in "${expected_outputs[@]}"; do
  if [[ ! -e "${KITTI_ROOT}/${name}" ]]; then
    echo "[ASAP][FAIL] Missing generated output: ${KITTI_ROOT}/${name}" >&2
    missing=1
  else
    echo "[ASAP][OK] Found generated output: ${KITTI_ROOT}/${name}"
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  fail "KITTI info generation did not produce all expected outputs."
fi

info "OpenPCDet KITTI info generation finished."
