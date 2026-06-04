#!/usr/bin/env bash
# Run ASAP with VPSDEPurifier (local-mean fallback) on all 3 attacked
# KITTI val sets, reusing the previously calibrated scorer JSONs.
set -euo pipefail

cd /root/autodl-tmp/ASAP
export PYTHONPATH=src:${PYTHONPATH:-}

PYTHON=/root/miniconda3/envs/asap/bin/python
LOGDIR=outputs/asap/logs
mkdir -p "${LOGDIR}"

run_one () {
    local atk="$1"
    local in_dir="outputs/attacks/kitti/${atk}/velodyne_attacked"
    local out_dir="outputs/asap_vpsde/kitti/${atk}/velodyne_purified"
    local meta="outputs/asap_vpsde/kitti/${atk}/meta.jsonl"
    local cfg="outputs/asap/configs/scorer_${atk}.json"
    local log="${LOGDIR}/asap_vpsde_${atk}.log"

    echo "[$(date +%H:%M:%S)] === ASAP vp_sde on ${atk} ==="
    "${PYTHON}" -m asap.pipeline \
        --input_dir "${in_dir}" \
        --out_dir "${out_dir}" \
        --meta_jsonl "${meta}" \
        --scorer_json "${cfg}" \
        --purifier vp_sde \
        --seed 42 \
        --workers 16 \
        2>&1 | tee "${log}"
    echo "[$(date +%H:%M:%S)] === Done ${atk} ==="
}

run_one E2.1_injection
run_one E2.2_perturbation
run_one E2.3_dropping

echo "[$(date +%H:%M:%S)] All ASAP vp_sde runs complete."
