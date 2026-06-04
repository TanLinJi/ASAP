#!/usr/bin/env bash
# Run ASAP with a trained local VP-SDE score-net checkpoint.
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/run_asap_vpsde_score_net.sh <score_net_ckpt> [attack_id]" >&2
    echo "Example: bash scripts/run_asap_vpsde_score_net.sh checkpoints/kitti/asap_score_net.pth E2.2_perturbation" >&2
    exit 1
fi

cd /root/autodl-tmp/ASAP
export PYTHONPATH=src:${PYTHONPATH:-}

CKPT="$1"
ONLY_ATTACK="${2:-}"
PYTHON=/root/miniconda3/envs/asap/bin/python
OUT_BASE="${OUT_BASE:-outputs/asap_vpsde_score_net}"
LOGDIR="${OUT_BASE}/logs"
WORKERS="${WORKERS:-2}"
SCORE_ANCHOR_BLEND="${SCORE_ANCHOR_BLEND:-0.0}"
SCORE_INJECTION_FILTER_RADIUS="${SCORE_INJECTION_FILTER_RADIUS:-0.0}"
SCORE_INJECTION_FILTER_MIN_NEIGHBORS="${SCORE_INJECTION_FILTER_MIN_NEIGHBORS:-0}"
mkdir -p "${LOGDIR}"

run_one () {
    local atk="$1"
    local in_dir="outputs/attacks/kitti/${atk}/velodyne_attacked"
    local out_dir="${OUT_BASE}/kitti/${atk}/velodyne_purified"
    local meta="${OUT_BASE}/kitti/${atk}/meta.jsonl"
    local cfg="outputs/asap/configs/scorer_${atk}.json"
    local log="${LOGDIR}/asap_vpsde_score_net_${atk}.log"

    echo "[$(date +%H:%M:%S)] === ASAP vp_sde score-net on ${atk} ==="
    "${PYTHON}" -m asap.pipeline \
        --input_dir "${in_dir}" \
        --out_dir "${out_dir}" \
        --meta_jsonl "${meta}" \
        --scorer_json "${cfg}" \
        --purifier vp_sde \
        --score_net_ckpt "${CKPT}" \
        --score_net_device cuda \
        --score_anchor_blend "${SCORE_ANCHOR_BLEND}" \
        --score_injection_filter_radius "${SCORE_INJECTION_FILTER_RADIUS}" \
        --score_injection_filter_min_neighbors "${SCORE_INJECTION_FILTER_MIN_NEIGHBORS}" \
        --t_star 0.05 \
        --seed 42 \
        --workers "${WORKERS}" \
        2>&1 | tee "${log}"
    echo "[$(date +%H:%M:%S)] === Done ${atk} ==="
}

if [ -n "${ONLY_ATTACK}" ]; then
    run_one "${ONLY_ATTACK}"
else
    run_one E2.1_injection
    run_one E2.2_perturbation
    run_one E2.3_dropping
fi

echo "[$(date +%H:%M:%S)] ASAP vp_sde score-net runs complete."
