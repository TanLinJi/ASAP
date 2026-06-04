#!/usr/bin/env bash
# eval_attacked_kitti.sh — Evaluate a KITTI detector on an attacked velodyne set.
#
# Usage:
#   bash scripts/eval_attacked_kitti.sh \
#       <attack_dir>       outputs/attacks/kitti/E2.1_injection \
#       <cfg_file>         cfgs/kitti_models/pointpillar.yaml \
#       <ckpt>             /root/autodl-tmp/ASAP/checkpoints/kitti/pointpillar_7728.pth \
#       <extra_tag>        E2.1_injection_pointpillars
#
# Optional env:
#   NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 DIST_PORT=18888 EVAL_WORKERS=4 bash ...
#
# The script:
#   1. Temporarily renames velodyne/ -> velodyne_clean_bak/
#   2. Symlinks velodyne_attacked/ -> velodyne/
#   3. Runs test.py
#   4. Restores the original velodyne/

set -euo pipefail

ATTACK_DIR="$1"
CFG_FILE="$2"
CKPT="$3"
EXTRA_TAG="$4"
BATCH_SIZE="${5:-4}"
NUM_GPUS="${NUM_GPUS:-1}"
if [ "${NUM_GPUS}" -gt 1 ]; then
    CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
else
    CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi
DIST_PORT="${DIST_PORT:-18888}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"

KITTI_TRAINING="/root/autodl-tmp/ASAP/data/kitti/training"
VELODYNE_ORIG="${KITTI_TRAINING}/velodyne"
VELODYNE_BAK="${KITTI_TRAINING}/velodyne_clean_bak"

# Support: dir containing velodyne_{attacked,defended,purified}/ OR raw .bin dir
if [ -d "${ATTACK_DIR}/velodyne_attacked" ]; then
    ATTACKED_DIR="$(realpath "${ATTACK_DIR}/velodyne_attacked")"
elif [ -d "${ATTACK_DIR}/velodyne_defended" ]; then
    ATTACKED_DIR="$(realpath "${ATTACK_DIR}/velodyne_defended")"
elif [ -d "${ATTACK_DIR}/velodyne_purified" ]; then
    ATTACKED_DIR="$(realpath "${ATTACK_DIR}/velodyne_purified")"
else
    ATTACKED_DIR="$(realpath "${ATTACK_DIR}")"
fi

# Safety check
if [ ! -d "${ATTACKED_DIR}" ]; then
    echo "ERROR: ${ATTACKED_DIR} does not exist" >&2
    exit 1
fi

# Backup original velodyne
if [ -L "${VELODYNE_ORIG}" ]; then
    # It's already a symlink; read the target
    ORIG_TARGET="$(readlink -f "${VELODYNE_ORIG}")"
    rm "${VELODYNE_ORIG}"
    echo "Backed up symlink: ${VELODYNE_ORIG} -> ${ORIG_TARGET}"
elif [ -d "${VELODYNE_ORIG}" ]; then
    mv "${VELODYNE_ORIG}" "${VELODYNE_BAK}"
    ORIG_TARGET=""
    echo "Backed up directory: ${VELODYNE_ORIG} -> ${VELODYNE_BAK}"
else
    echo "ERROR: ${VELODYNE_ORIG} not found" >&2
    exit 1
fi

# Symlink attacked set
ln -s "${ATTACKED_DIR}" "${VELODYNE_ORIG}"
echo "Symlinked: ${VELODYNE_ORIG} -> ${ATTACKED_DIR}"

# Restore function (runs on exit, even on error)
restore() {
    rm -f "${VELODYNE_ORIG}"
    if [ -n "${ORIG_TARGET:-}" ]; then
        ln -s "${ORIG_TARGET}" "${VELODYNE_ORIG}"
        echo "Restored symlink: ${VELODYNE_ORIG} -> ${ORIG_TARGET}"
    elif [ -d "${VELODYNE_BAK}" ]; then
        mv "${VELODYNE_BAK}" "${VELODYNE_ORIG}"
        echo "Restored directory: ${VELODYNE_BAK} -> ${VELODYNE_ORIG}"
    fi
}
trap restore EXIT

# Run evaluation
cd /root/autodl-tmp/ASAP/third_party/OpenPCDet/tools

if [ "${NUM_GPUS}" -gt 1 ]; then
    env \
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
        CUDA_HOME=/usr/local/cuda-12.4 \
        LD_LIBRARY_PATH=/root/miniconda3/envs/asap/lib/python3.9/site-packages/torch/lib:/usr/local/cuda-12.4/lib64 \
        /root/miniconda3/envs/asap/bin/python -m torch.distributed.run \
            --nproc_per_node="${NUM_GPUS}" \
            --master_port="${DIST_PORT}" \
            test.py \
            --cfg_file "${CFG_FILE}" \
            --batch_size "${BATCH_SIZE}" \
            --workers "${EVAL_WORKERS}" \
            --ckpt "${CKPT}" \
            --extra_tag "${EXTRA_TAG}" \
            --launcher pytorch \
            --tcp_port "${DIST_PORT}"
else
    env \
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
        CUDA_HOME=/usr/local/cuda-12.4 \
        LD_LIBRARY_PATH=/root/miniconda3/envs/asap/lib/python3.9/site-packages/torch/lib:/usr/local/cuda-12.4/lib64 \
        /root/miniconda3/envs/asap/bin/python -u test.py \
            --cfg_file "${CFG_FILE}" \
            --batch_size "${BATCH_SIZE}" \
            --workers "${EVAL_WORKERS}" \
            --ckpt "${CKPT}" \
            --extra_tag "${EXTRA_TAG}"
fi

echo "Done: ${EXTRA_TAG}"
