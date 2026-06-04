#!/usr/bin/env bash
# Run no-defense nuScenes evals for accepted attack budgets only.
#
# Accepted after the VoxelNeXt budget gate:
#   - Injection: original E2.1 keyframe-only budget
#   - Perturbation: budget-gated epsilon=0.5
#
# Dropping is intentionally excluded because keyframe-only drop_ratio=1.0 still
# has weak VoxelNeXt ASR under MAX_SWEEPS=10.

set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/ASAP}"
ATTACK_ROOT="${ATTACK_ROOT:-${ROOT}/outputs/attacks/nuscenes_keyframe}"
BUDGET_ROOT="${BUDGET_ROOT:-${ROOT}/outputs/attacks/nuscenes_keyframe_budget_gate}"
DETECTOR="${DETECTOR:-transfusion_lidar}"
NUM_GPUS="${NUM_GPUS:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
DIST_PORT_BASE="${DIST_PORT_BASE:-19500}"

case "${DETECTOR}" in
    voxelnext)
        CFG="cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"
        CKPT="${ROOT}/checkpoints/nuscenes/voxelnext_nuscenes_kernel1.pth"
        MODEL="cbgs_voxel0075_voxelnext"
        TAG_PREFIX="E4_3_voxelnext_accepted_no_defense"
        OUT_JSON="${BUDGET_ROOT}/nuscenes_voxelnext_accepted_no_defense_eval_summary.json"
        CLEAN_MAP="0.605195155366683"
        CLEAN_NDS="0.6664"
        ;;
    transfusion|transfusion_lidar)
        CFG="cfgs/nuscenes_models/transfusion_lidar.yaml"
        CKPT="${ROOT}/checkpoints/nuscenes/cbgs_transfusion_lidar.pth"
        MODEL="transfusion_lidar"
        TAG_PREFIX="E4_4_transfusion_lidar_accepted_no_defense"
        OUT_JSON="${BUDGET_ROOT}/nuscenes_transfusion_lidar_accepted_no_defense_eval_summary.json"
        CLEAN_MAP="0.645704360834918"
        CLEAN_NDS="0.6943"
        ;;
    *)
        echo "ERROR: DETECTOR must be voxelnext or transfusion_lidar, got ${DETECTOR}" >&2
        exit 2
        ;;
esac

cd "${ROOT}"

has_metrics() {
    local tag="$1"
    find "third_party/OpenPCDet/output/nuscenes_models/${MODEL}/${tag}/eval" \
        -name 'metrics_summary.json' -size +0c -print -quit 2>/dev/null | grep -q .
}

run_one() {
    local label="$1"
    local attack_dir="$2"
    local tag_suffix="$3"
    local port="$4"
    local tag="${TAG_PREFIX}_${tag_suffix}"

    if [ ! -d "${attack_dir}/samples/LIDAR_TOP" ]; then
        echo "ERROR: missing attack set: ${attack_dir}/samples/LIDAR_TOP" >&2
        exit 1
    fi

    if [ -z "${FORCE_EVAL:-}" ] && has_metrics "${tag}"; then
        echo "[skip] ${tag} already has metrics_summary.json"
        return
    fi

    echo "[run] ${DETECTOR} ${label} -> ${tag}"
    NUM_GPUS="${NUM_GPUS}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    EVAL_WORKERS="${EVAL_WORKERS}" \
    DIST_PORT="${port}" \
        bash scripts/eval_attacked_nuscenes.sh \
            "${attack_dir}" \
            "${CFG}" \
            "${CKPT}" \
            "${tag}"
}

INJ_DIR="${ATTACK_ROOT}/E2.1_injection"
PERT_DIR="${BUDGET_ROOT}/E2.2_perturbation_eps0p5"
INJ_TAG="${TAG_PREFIX}_E2_1"
PERT_TAG="${TAG_PREFIX}_E2_2_eps0p5"

run_one "E2.1 injection" "${INJ_DIR}" "E2_1" "$((DIST_PORT_BASE + 1))"
run_one "E2.2 perturbation epsilon=0.5" "${PERT_DIR}" "E2_2_eps0p5" "$((DIST_PORT_BASE + 2))"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] skip metrics parser because no eval metrics are produced"
    exit 0
fi

/root/miniconda3/envs/asap/bin/python scripts/parse_nuscenes_eval_summary.py \
    --model "${MODEL}" \
    --clean-map "${CLEAN_MAP}" \
    --clean-nds "${CLEAN_NDS}" \
    --out "${OUT_JSON}" \
    --entry "E2.1 no-defense:${INJ_TAG}" \
    --entry "E2.2 eps0.5 no-defense:${PERT_TAG}"

echo "[done] wrote ${OUT_JSON}"
