#!/usr/bin/env bash
# Run ASAP purification on all 3 attacked KITTI val sets, sequentially.
set -euo pipefail

cd /root/autodl-tmp/ASAP
export PYTHONPATH=src:${PYTHONPATH:-}

PYTHON=/root/miniconda3/envs/asap/bin/python
LOGDIR=outputs/asap/logs
mkdir -p "${LOGDIR}" outputs/asap/configs

run_one () {
    local atk="$1"
    local in_dir="outputs/attacks/kitti/${atk}/velodyne_attacked"
    local out_dir="outputs/asap/kitti/${atk}/velodyne_purified"
    local meta="outputs/asap/kitti/${atk}/meta.jsonl"
    local cfg="outputs/asap/configs/scorer_${atk}.json"
    local log="${LOGDIR}/asap_${atk}.log"

    echo "[$(date +%H:%M:%S)] === Starting ASAP on ${atk} ==="
    "${PYTHON}" -m asap.pipeline \
        --input_dir "${in_dir}" \
        --out_dir "${out_dir}" \
        --meta_jsonl "${meta}" \
        --calibrate_with_clean_dir data/kitti/training/velodyne \
        --calibrate_n_frames 20 \
        --save_scorer_json "${cfg}" \
        --purifier drop \
        --seed 42 \
        --workers 16 \
        2>&1 | tee "${log}"
    echo "[$(date +%H:%M:%S)] === Done ${atk} ==="
}

run_one E2.1_injection
run_one E2.2_perturbation
run_one E2.3_dropping

echo "[$(date +%H:%M:%S)] All ASAP runs complete."
