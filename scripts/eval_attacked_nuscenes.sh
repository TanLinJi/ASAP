#!/usr/bin/env bash
# Evaluate an OpenPCDet nuScenes detector on a keyframe-attacked LIDAR_TOP set.
#
# Usage:
#   NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash scripts/eval_attacked_nuscenes.sh \
#       outputs/attacks/nuscenes_keyframe/E2.1_injection \
#       cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml \
#       /root/autodl-tmp/ASAP/checkpoints/nuscenes/voxelnext_nuscenes_kernel1.pth \
#       E4_3_voxelnext_keyframe_no_defense_E2_1
#
# The script temporarily swaps:
#   data/nuscenes/<version>/samples/LIDAR_TOP
# with the attacked directory, runs OpenPCDet eval, then restores the clean
# directory via trap. A file lock prevents concurrent nuScenes eval swaps.

set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <attack_dir> <cfg_file> <ckpt> <extra_tag> [batch_size]" >&2
    exit 2
fi

ROOT="${ROOT:-/root/autodl-tmp/ASAP}"
ATTACK_DIR="$1"
CFG_FILE="$2"
CKPT="$3"
EXTRA_TAG="$4"
NUM_GPUS="${NUM_GPUS:-1}"
if [ "$#" -ge 5 ]; then
    BATCH_SIZE="$5"
else
    BATCH_SIZE="${BATCH_SIZE:-${NUM_GPUS}}"
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
    CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
else
    CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

DIST_PORT="${DIST_PORT:-18888}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"
NUSC_VERSION="${NUSC_VERSION:-v1.0-trainval}"
EVAL_TAG="${EVAL_TAG:-${NUSC_VERSION}}"
NUSC_ROOT="${NUSC_ROOT:-${ROOT}/data/nuscenes/${NUSC_VERSION}}"
LOCK_FILE="${LOCK_FILE:-${ROOT}/data/nuscenes/.lidar_top_eval.lock}"
PYTHON="${PYTHON:-/root/miniconda3/envs/asap/bin/python}"

if [ "${NUM_GPUS}" -gt 1 ] && [ $((BATCH_SIZE % NUM_GPUS)) -ne 0 ]; then
    echo "ERROR: batch size (${BATCH_SIZE}) must be divisible by NUM_GPUS (${NUM_GPUS})" >&2
    exit 2
fi

if [ -d "${ATTACK_DIR}/samples/LIDAR_TOP" ]; then
    ATTACKED_LIDAR="$(realpath "${ATTACK_DIR}/samples/LIDAR_TOP")"
elif [ "$(basename "${ATTACK_DIR}")" = "LIDAR_TOP" ] && [ -d "${ATTACK_DIR}" ]; then
    ATTACKED_LIDAR="$(realpath "${ATTACK_DIR}")"
else
    echo "ERROR: cannot find attacked samples/LIDAR_TOP under ${ATTACK_DIR}" >&2
    exit 1
fi

LIDAR_ORIG="${NUSC_ROOT}/samples/LIDAR_TOP"
LIDAR_BAK="${NUSC_ROOT}/samples/LIDAR_TOP_clean_bak_eval_$$"
ORIG_TARGET=""
ORIG_WAS_SYMLINK=0
SWAPPED=0

restore() {
    set +e
    if [ "${SWAPPED}" -eq 1 ]; then
        rm -f "${LIDAR_ORIG}"
        if [ "${ORIG_WAS_SYMLINK}" -eq 1 ]; then
            ln -s "${ORIG_TARGET}" "${LIDAR_ORIG}"
            echo "[restore] ${LIDAR_ORIG} -> ${ORIG_TARGET}"
        elif [ -d "${LIDAR_BAK}" ]; then
            mv "${LIDAR_BAK}" "${LIDAR_ORIG}"
            echo "[restore] ${LIDAR_BAK} -> ${LIDAR_ORIG}"
        fi
    fi
}
trap restore EXIT

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "ERROR: another nuScenes LIDAR_TOP swap/eval is running; lock=${LOCK_FILE}" >&2
    exit 1
fi

if [ -L "${LIDAR_ORIG}" ]; then
    ORIG_TARGET="$(readlink -f "${LIDAR_ORIG}")"
    ORIG_WAS_SYMLINK=1
    rm "${LIDAR_ORIG}"
    echo "[backup] symlink ${LIDAR_ORIG} -> ${ORIG_TARGET}"
elif [ -d "${LIDAR_ORIG}" ]; then
    mv "${LIDAR_ORIG}" "${LIDAR_BAK}"
    echo "[backup] directory ${LIDAR_ORIG} -> ${LIDAR_BAK}"
else
    echo "ERROR: original LIDAR_TOP not found: ${LIDAR_ORIG}" >&2
    exit 1
fi

ln -s "${ATTACKED_LIDAR}" "${LIDAR_ORIG}"
SWAPPED=1
echo "[swap] ${LIDAR_ORIG} -> ${ATTACKED_LIDAR}"

cd "${ROOT}/third_party/OpenPCDet/tools"

COMMON_ENV=(
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
    CUDA_HOME=/usr/local/cuda-12.4
    LD_LIBRARY_PATH=/root/miniconda3/envs/asap/lib/python3.9/site-packages/torch/lib:/usr/local/cuda-12.4/lib64
)

COMMON_ARGS=(
    --cfg_file "${CFG_FILE}"
    --batch_size "${BATCH_SIZE}"
    --workers "${EVAL_WORKERS}"
    --ckpt "${CKPT}"
    --extra_tag "${EXTRA_TAG}"
    --eval_tag "${EVAL_TAG}"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] cd ${ROOT}/third_party/OpenPCDet/tools"
    echo "[dry-run] env ${COMMON_ENV[*]} ${PYTHON} -u test.py ${COMMON_ARGS[*]}"
    exit 0
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
    env "${COMMON_ENV[@]}" \
        "${PYTHON}" -m torch.distributed.run \
            --nproc_per_node="${NUM_GPUS}" \
            --master_port="${DIST_PORT}" \
            test.py \
            "${COMMON_ARGS[@]}" \
            --launcher pytorch \
            --tcp_port "${DIST_PORT}"
else
    env "${COMMON_ENV[@]}" \
        "${PYTHON}" -u test.py \
            "${COMMON_ARGS[@]}"
fi

echo "[done] ${EXTRA_TAG}"
