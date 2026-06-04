#!/usr/bin/env bash
# Generate and evaluate stronger nuScenes keyframe-only attack budgets.
#
# This is a VoxelNeXt-only gate used before spending GPU time on E4.4 or
# nuScenes defense matrices. It keeps candidate outputs separate from the
# original E2.x directories so the accepted paper protocol remains auditable.
#
# Recommended:
#   tmux new-session -d -s asap_nusc_budget_gate_$(date +%Y%m%d_%H%M%S) \
#     "cd /root/autodl-tmp/ASAP && NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 \
#      bash scripts/run_nuscenes_attack_budget_gate.sh 2>&1 | tee \
#      outputs/attacks/nuscenes_keyframe_budget_gate/asap_nusc_budget_gate_$(date +%Y%m%d_%H%M%S).log"

set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/ASAP}"
INFO_PKL="${INFO_PKL:-${ROOT}/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl}"
NUSC_ROOT="${NUSC_ROOT:-${ROOT}/data/nuscenes/v1.0-trainval}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/attacks/nuscenes_keyframe_budget_gate}"
PYTHON="${PYTHON:-/root/miniconda3/envs/asap/bin/python}"
NUM_GPUS="${NUM_GPUS:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
DIST_PORT_BASE="${DIST_PORT_BASE:-19400}"

PERT_EPSILON="${PERT_EPSILON:-0.5}"
DROP_RATIO_STRONG="${DROP_RATIO_STRONG:-1.0}"
MAX_FRAMES="${MAX_FRAMES:-}"

CFG="cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"
CKPT="${ROOT}/checkpoints/nuscenes/voxelnext_nuscenes_kernel1.pth"
MODEL="cbgs_voxel0075_voxelnext"
CLEAN_MAP="0.605195155366683"
CLEAN_NDS="0.6664"
SUMMARY_JSON="${OUT_ROOT}/nuscenes_voxelnext_budget_gate_eval_summary.json"

cd "${ROOT}"
mkdir -p "${OUT_ROOT}"

common_attack_args=(
    --info_pkl "${INFO_PKL}"
    --nuscenes_root "${NUSC_ROOT}"
)
if [ -n "${MAX_FRAMES}" ]; then
    common_attack_args+=(--max_frames "${MAX_FRAMES}")
fi

run_attack() {
    local attack_id="$1"
    local attack="$2"
    shift 2
    local out_dir="${OUT_ROOT}/${attack_id}"

    if [ -f "${out_dir}/attack_meta.jsonl" ] && [ -z "${FORCE_ATTACK:-}" ]; then
        echo "[skip-attack] ${attack_id} already has attack_meta.jsonl"
        return
    fi

    echo "[run-attack] ${attack_id}"
    PYTHONPATH=src "${PYTHON}" -m asap.attacks.nuscenes_keyframe_attack \
        "${common_attack_args[@]}" \
        --out_dir "${out_dir}" \
        --attack "${attack}" \
        "$@"
}

has_metrics() {
    local tag="$1"
    find "third_party/OpenPCDet/output/nuscenes_models/${MODEL}/${tag}/eval" \
        -name 'metrics_summary.json' -size +0c -print -quit 2>/dev/null | grep -q .
}

run_eval() {
    local attack_id="$1"
    local tag="$2"
    local port="$3"
    local attack_dir="${OUT_ROOT}/${attack_id}"

    if [ ! -d "${attack_dir}/samples/LIDAR_TOP" ]; then
        echo "ERROR: missing attack set: ${attack_dir}/samples/LIDAR_TOP" >&2
        exit 1
    fi

    if [ -z "${FORCE_EVAL:-}" ] && has_metrics "${tag}"; then
        echo "[skip-eval] ${tag} already has metrics_summary.json"
        return
    fi

    echo "[run-eval] voxelnext ${attack_id} -> ${tag}"
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

sanitize_float() {
    local value="$1"
    value="${value//./p}"
    value="${value//-/_}"
    echo "${value}"
}

PERT_SUFFIX="eps$(sanitize_float "${PERT_EPSILON}")"
DROP_SUFFIX="drop$(sanitize_float "${DROP_RATIO_STRONG}")"
if [ -n "${MAX_FRAMES}" ]; then
    PERT_SUFFIX="${PERT_SUFFIX}_max${MAX_FRAMES}"
    DROP_SUFFIX="${DROP_SUFFIX}_max${MAX_FRAMES}"
fi

PERT_ID="E2.2_perturbation_${PERT_SUFFIX}"
PERT_TAG="E4_3_voxelnext_budget_gate_E2_2_${PERT_SUFFIX}"
DROP_ID="E2.3_dropping_${DROP_SUFFIX}"
DROP_TAG="E4_3_voxelnext_budget_gate_E2_3_${DROP_SUFFIX}"

run_attack "${PERT_ID}" perturbation --epsilon "${PERT_EPSILON}" --seed 42
run_attack "${DROP_ID}" dropping --drop_ratio "${DROP_RATIO_STRONG}" --seed 42

run_eval "${PERT_ID}" "${PERT_TAG}" "$((DIST_PORT_BASE + 1))"
run_eval "${DROP_ID}" "${DROP_TAG}" "$((DIST_PORT_BASE + 2))"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] skip metrics parser because no eval metrics are produced"
    exit 0
fi

"${PYTHON}" scripts/parse_nuscenes_eval_summary.py \
    --model "${MODEL}" \
    --clean-map "${CLEAN_MAP}" \
    --clean-nds "${CLEAN_NDS}" \
    --out "${SUMMARY_JSON}" \
    --entry "E2.2 epsilon=${PERT_EPSILON}:${PERT_TAG}" \
    --entry "E2.3 drop_ratio=${DROP_RATIO_STRONG}:${DROP_TAG}"

echo "[done] wrote ${SUMMARY_JSON}"
