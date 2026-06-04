#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ASAP
export PYTHONPATH=src:${PYTHONPATH:-}

PYTHON=/root/miniconda3/envs/asap/bin/python
LOGDIR=outputs/asap_hybrid_conditional/logs
mkdir -p "${LOGDIR}"

run_one () {
    local atk="$1"
    local in_dir="outputs/attacks/kitti/${atk}/velodyne_attacked"
    local out_dir="outputs/asap_hybrid_conditional/kitti/${atk}/velodyne_purified"
    local meta="outputs/asap_hybrid_conditional/kitti/${atk}/meta.jsonl"
    local cfg="outputs/asap/configs/scorer_${atk}.json"
    local log="${LOGDIR}/asap_hybrid_conditional_${atk}.log"

    echo "[$(date +%H:%M:%S)] === ASAP hybrid-conditional on ${atk} ==="
    "${PYTHON}" -m asap.pipeline \
        --input_dir "${in_dir}" \
        --out_dir "${out_dir}" \
        --meta_jsonl "${meta}" \
        --scorer_json "${cfg}" \
        --purifier hybrid \
        --seed 42 \
        --workers 16 \
        2>&1 | tee "${log}"
    echo "[$(date +%H:%M:%S)] === Done ${atk} ==="
}

run_one E2.1_injection
run_one E2.2_perturbation
run_one E2.3_dropping

echo "[$(date +%H:%M:%S)] All ASAP hybrid-conditional runs complete."
