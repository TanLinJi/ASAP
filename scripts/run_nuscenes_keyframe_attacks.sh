#!/usr/bin/env bash
# Generate keyframe-only nuScenes attack sets for E2.x@nuScenes.

set -euo pipefail

ROOT="/root/autodl-tmp/ASAP"
INFO_PKL="${INFO_PKL:-${ROOT}/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl}"
NUSC_ROOT="${NUSC_ROOT:-${ROOT}/data/nuscenes/v1.0-trainval}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/attacks/nuscenes_keyframe}"
MAX_FRAMES="${MAX_FRAMES:-}"
PYTHON="${PYTHON:-/root/miniconda3/envs/asap/bin/python}"

cd "${ROOT}"

common_args=(
    --info_pkl "${INFO_PKL}"
    --nuscenes_root "${NUSC_ROOT}"
)
if [ -n "${MAX_FRAMES}" ]; then
    common_args+=(--max_frames "${MAX_FRAMES}")
fi

run_one() {
    local attack_id="$1"
    local attack="$2"
    shift 2
    local out_dir="${OUT_ROOT}/${attack_id}"
    if [ -f "${out_dir}/attack_meta.jsonl" ] && [ -z "${FORCE:-}" ]; then
        echo "[skip] ${attack_id} already has attack_meta.jsonl"
        return
    fi
    echo "[run] ${attack_id}"
    PYTHONPATH=src "${PYTHON}" -m asap.attacks.nuscenes_keyframe_attack \
        "${common_args[@]}" \
        --out_dir "${out_dir}" \
        --attack "${attack}" \
        "$@"
}

run_one E2.1_injection injection --k_inj "${K_INJ:-50}" --r_inj "${R_INJ:-0.5}" --seed 42
run_one E2.2_perturbation perturbation --epsilon "${EPSILON:-0.3}" --seed 42
run_one E2.3_dropping dropping --drop_ratio "${DROP_RATIO:-0.5}" --seed 42

echo "[done] keyframe attacks -> ${OUT_ROOT}"
