#!/usr/bin/env bash
# Prepare nuScenes-specific ASAP Track A assets.
#
# Outputs:
#   - checkpoints/nuscenes/asap_score_net_trackA_v1.pth
#   - outputs/asap_nuscenes_trackA_assets/configs/scorer_E2_1.json
#   - outputs/asap_nuscenes_trackA_assets/configs/scorer_E2_2_eps0p5.json
#
# Default run is sized as a starter asset build. Increase TRAIN_MAX_FRAMES /
# CALIBRATE_N_FRAMES only after smoke passes.
#
# Disconnect-safe example:
#   tmux new -s asap_nusc_tracka_assets_$(date +%Y%m%d_%H%M%S) \
#     'cd /root/autodl-tmp/ASAP && CUDA_VISIBLE_DEVICES=0,1 bash scripts/prepare_nuscenes_tracka_assets.sh'
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/ASAP}"
PYTHON="${PYTHON:-/root/miniconda3/envs/asap/bin/python}"
CLEAN_DIR="${CLEAN_DIR:-${ROOT}/data/nuscenes/v1.0-trainval/samples/LIDAR_TOP}"
ATTACK_ROOT="${ATTACK_ROOT:-${ROOT}/outputs/attacks/nuscenes_keyframe}"
BUDGET_ROOT="${BUDGET_ROOT:-${ROOT}/outputs/attacks/nuscenes_keyframe_budget_gate}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/asap_nuscenes_trackA_assets}"
CFG_DIR="${CFG_DIR:-${OUT_ROOT}/configs}"
CKPT="${CKPT:-${ROOT}/checkpoints/nuscenes/asap_score_net_trackA_v1.pth}"
LOGDIR="${LOGDIR:-${OUT_ROOT}/logs}"

TRAIN_MAX_FRAMES="${TRAIN_MAX_FRAMES:-128}"
PATCHES_PER_FRAME="${PATCHES_PER_FRAME:-64}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
DATA_PARALLEL="${DATA_PARALLEL:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

CALIBRATE_N_FRAMES="${CALIBRATE_N_FRAMES:-40}"
FORCE_CALIBRATE="${FORCE_CALIBRATE:-0}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
mkdir -p "${CFG_DIR}" "${LOGDIR}" "$(dirname "${CKPT}")"
cd "${ROOT}"

require_dir() {
    if [ ! -d "$1" ]; then
        echo "ERROR: missing directory: $1" >&2
        exit 1
    fi
}

run_cmd() {
    local log="$1"
    shift
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry-run] $*"
        return
    fi
    echo "[run] $*" | tee "${log}"
    "$@" 2>&1 | tee -a "${log}"
}

require_dir "${CLEAN_DIR}"
require_dir "${ATTACK_ROOT}/E2.1_injection/samples/LIDAR_TOP"
require_dir "${BUDGET_ROOT}/E2.2_perturbation_eps0p5/samples/LIDAR_TOP"

if [ "${DRY_RUN}" = "1" ]; then
    echo "[dry-run] ROOT=${ROOT}"
    echo "[dry-run] CLEAN_DIR=${CLEAN_DIR}"
    echo "[dry-run] CKPT=${CKPT}"
    echo "[dry-run] CFG_DIR=${CFG_DIR}"
fi

if [ "${FORCE_TRAIN}" = "1" ] || [ ! -s "${CKPT}" ]; then
    train_args=(
        "${PYTHON}" scripts/train_vpsde_score_net.py
        --clean_dir "${CLEAN_DIR}"
        --out_ckpt "${CKPT}"
        --max_frames "${TRAIN_MAX_FRAMES}"
        --patches_per_frame "${PATCHES_PER_FRAME}"
        --epochs "${EPOCHS}"
        --batch_size "${BATCH_SIZE}"
        --num_features 5
        --device "${TRAIN_DEVICE}"
    )
    if [ "${DATA_PARALLEL}" = "1" ]; then
        train_args+=(--data_parallel)
    fi
    run_cmd "${LOGDIR}/train_score_net.log" "${train_args[@]}"
else
    echo "[skip] score-net checkpoint exists: ${CKPT}"
fi

calibrate_one() {
    local suffix="$1"
    local attack_dir="$2"
    local out_json="${CFG_DIR}/scorer_${suffix}.json"
    local log="${LOGDIR}/calibrate_${suffix}.log"

    if [ "${FORCE_CALIBRATE}" != "1" ] && [ -s "${out_json}" ]; then
        echo "[skip] scorer exists: ${out_json}"
        return
    fi

    run_cmd "${log}" \
        "${PYTHON}" -m asap.pipeline \
            --input_dir "${attack_dir}" \
            --out_dir "${OUT_ROOT}/calibrate_dummy_${suffix}" \
            --calibrate_with_clean_dir "${CLEAN_DIR}" \
            --calibrate_n_frames "${CALIBRATE_N_FRAMES}" \
            --save_scorer_json "${out_json}" \
            --calibrate_only \
            --num_features 5 \
            --purifier vp_sde
}

calibrate_one "E2_1" "${ATTACK_ROOT}/E2.1_injection/samples/LIDAR_TOP"
calibrate_one "E2_2_eps0p5" "${BUDGET_ROOT}/E2.2_perturbation_eps0p5/samples/LIDAR_TOP"

echo "[done] nuScenes Track A assets ready"
echo "  CKPT=${CKPT}"
echo "  CFG_DIR=${CFG_DIR}"
