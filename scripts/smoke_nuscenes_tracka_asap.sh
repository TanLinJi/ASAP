#!/usr/bin/env bash
# Smoke-test nuScenes Track A ASAP assets on a tiny accepted-attack subset.
#
# Runs Injection and Perturbation on separate GPUs when two GPUs are visible.
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/ASAP}"
PYTHON="${PYTHON:-/root/miniconda3/envs/asap/bin/python}"
ATTACK_ROOT="${ATTACK_ROOT:-${ROOT}/outputs/attacks/nuscenes_keyframe}"
BUDGET_ROOT="${BUDGET_ROOT:-${ROOT}/outputs/attacks/nuscenes_keyframe_budget_gate}"
ASSET_ROOT="${ASSET_ROOT:-${ROOT}/outputs/asap_nuscenes_trackA_assets}"
CFG_DIR="${CFG_DIR:-${ASSET_ROOT}/configs}"
CKPT="${CKPT:-${ROOT}/checkpoints/nuscenes/asap_score_net_trackA_v1.pth}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/smoke/nuscenes_tracka_asap}"
SMOKE_FRAMES="${SMOKE_FRAMES:-2}"
GPUS="${GPUS:-0,1}"
DRY_RUN="${DRY_RUN:-0}"

ASAP_SCORE_ANCHOR_BLEND="${ASAP_SCORE_ANCHOR_BLEND:-0.75}"
ASAP_SCORE_INJECTION_FILTER_RADIUS="${ASAP_SCORE_INJECTION_FILTER_RADIUS:-0.4}"
ASAP_SCORE_INJECTION_FILTER_MIN_NEIGHBORS="${ASAP_SCORE_INJECTION_FILTER_MIN_NEIGHBORS:-5}"
ASAP_SCORE_FILTER_ATTACK_KINDS="${ASAP_SCORE_FILTER_ATTACK_KINDS:-injection}"
ASAP_T_STAR="${ASAP_T_STAR:-0.05}"
ASAP_SCORE_STEP_SIZE="${ASAP_SCORE_STEP_SIZE:-1.0}"
ASAP_SCORE_CLIP="${ASAP_SCORE_CLIP:-3.0}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}/logs"
cd "${ROOT}"

require_file() {
    if [ ! -s "$1" ]; then
        echo "ERROR: missing file: $1" >&2
        exit 1
    fi
}

require_dir() {
    if [ ! -d "$1" ]; then
        echo "ERROR: missing directory: $1" >&2
        exit 1
    fi
}

require_file "${CKPT}"
require_file "${CFG_DIR}/scorer_E2_1.json"
require_file "${CFG_DIR}/scorer_E2_2_eps0p5.json"
require_dir "${ATTACK_ROOT}/E2.1_injection/samples/LIDAR_TOP"
require_dir "${BUDGET_ROOT}/E2.2_perturbation_eps0p5/samples/LIDAR_TOP"

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
GPU0="${GPU_IDS[0]:-0}"
GPU1="${GPU_IDS[1]:-${GPU0}}"

prepare_subset() {
    local src="$1"
    local dst="$2"
    rm -rf "${dst}"
    mkdir -p "${dst}"
    local n=0
    while IFS= read -r f; do
        ln -sf "$(realpath "${f}")" "${dst}/$(basename "${f}")"
        n=$((n + 1))
        if [ "${n}" -ge "${SMOKE_FRAMES}" ]; then
            break
        fi
    done < <(find "${src}" -maxdepth 1 -name '*.bin' | sort)
    if [ "${n}" -ne "${SMOKE_FRAMES}" ]; then
        echo "ERROR: expected ${SMOKE_FRAMES} frames in ${src}, found ${n}" >&2
        exit 1
    fi
}

run_one() {
    local attack_label="$1"
    local attack_kind="$2"
    local scorer="$3"
    local input_dir="$4"
    local gpu="$5"
    local out_dir="${OUT_ROOT}/${attack_label}/samples/LIDAR_TOP"
    local meta="${OUT_ROOT}/${attack_label}/meta.jsonl"
    local log="${OUT_ROOT}/logs/${attack_label}_gpu${gpu}.log"

    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"
    rm -f "${meta}"

    local cmd=(
        "${PYTHON}" -m asap.pipeline
        --input_dir "${input_dir}"
        --out_dir "${out_dir}"
        --meta_jsonl "${meta}"
        --scorer_json "${scorer}"
        --purifier vp_sde
        --score_net_ckpt "${CKPT}"
        --score_net_device cuda
        --score_step_size "${ASAP_SCORE_STEP_SIZE}"
        --score_clip "${ASAP_SCORE_CLIP}"
        --score_anchor_blend "${ASAP_SCORE_ANCHOR_BLEND}"
        --score_injection_filter_radius "${ASAP_SCORE_INJECTION_FILTER_RADIUS}"
        --score_injection_filter_min_neighbors "${ASAP_SCORE_INJECTION_FILTER_MIN_NEIGHBORS}"
        --score_filter_attack_kinds "${ASAP_SCORE_FILTER_ATTACK_KINDS}"
        --t_star "${ASAP_T_STAR}"
        --attack_kind "${attack_kind}"
        --seed 42
        --num_features 5
        --workers 1
    )
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry-run] CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
        return
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${log}" 2>&1
}

validate_one() {
    local attack_label="$1"
    local out_dir="${OUT_ROOT}/${attack_label}/samples/LIDAR_TOP"
    local meta="${OUT_ROOT}/${attack_label}/meta.jsonl"
    local produced
    local meta_lines
    produced="$(find "${out_dir}" -maxdepth 1 -name '*.bin' | wc -l)"
    meta_lines="$(wc -l < "${meta}")"
    if [ "${produced}" -ne "${SMOKE_FRAMES}" ] || [ "${meta_lines}" -ne "${SMOKE_FRAMES}" ]; then
        echo "ERROR: ${attack_label} produced=${produced}, meta_lines=${meta_lines}, expected=${SMOKE_FRAMES}" >&2
        exit 1
    fi
    "${PYTHON}" - "$out_dir" <<'PY'
import sys
from pathlib import Path
import numpy as np

out_dir = Path(sys.argv[1])
for path in sorted(out_dir.glob("*.bin")):
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size == 0 or arr.size % 5 != 0:
        raise SystemExit(f"invalid 5D point cloud: {path} size={arr.size}")
print("validated", out_dir)
PY
}

INJ_SUBSET="${OUT_ROOT}/subsets/E2_1_injection"
PERT_SUBSET="${OUT_ROOT}/subsets/E2_2_eps0p5"
prepare_subset "${ATTACK_ROOT}/E2.1_injection/samples/LIDAR_TOP" "${INJ_SUBSET}"
prepare_subset "${BUDGET_ROOT}/E2.2_perturbation_eps0p5/samples/LIDAR_TOP" "${PERT_SUBSET}"

if [ "${DRY_RUN}" = "1" ]; then
    run_one "E2_1_injection" "injection" "${CFG_DIR}/scorer_E2_1.json" "${INJ_SUBSET}" "${GPU0}"
    run_one "E2_2_eps0p5" "perturbation" "${CFG_DIR}/scorer_E2_2_eps0p5.json" "${PERT_SUBSET}" "${GPU1}"
    echo "[dry-run] smoke command generation complete"
    exit 0
fi

run_one "E2_1_injection" "injection" "${CFG_DIR}/scorer_E2_1.json" "${INJ_SUBSET}" "${GPU0}" &
PID0="$!"
run_one "E2_2_eps0p5" "perturbation" "${CFG_DIR}/scorer_E2_2_eps0p5.json" "${PERT_SUBSET}" "${GPU1}" &
PID1="$!"

FAILED=0
wait "${PID0}" || FAILED=1
wait "${PID1}" || FAILED=1
if [ "${FAILED}" -ne 0 ]; then
    echo "ERROR: at least one smoke purification failed. Logs: ${OUT_ROOT}/logs" >&2
    exit 1
fi

validate_one "E2_1_injection"
validate_one "E2_2_eps0p5"

echo "[done] nuScenes Track A ASAP smoke passed: ${OUT_ROOT}"
