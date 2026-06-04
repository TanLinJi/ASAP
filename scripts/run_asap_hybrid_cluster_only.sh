#!/usr/bin/env bash
# Hybrid M3 ablation: cluster-drop branch only (isolation drop disabled).
# Used to verify the §7.3 diagnosis that the isolation-radius drop is
# the cause of the Perturbation regression vs ASAP-vpsde.
set -euo pipefail

cd /root/autodl-tmp/ASAP
export PYTHONPATH=src:${PYTHONPATH:-}

PYTHON=/root/miniconda3/envs/asap/bin/python
LOGDIR=outputs/asap_hybrid_cluster_only/logs
mkdir -p "${LOGDIR}"

run_one () {
    local atk="$1"
    local in_dir="outputs/attacks/kitti/${atk}/velodyne_attacked"
    local out_dir="outputs/asap_hybrid_cluster_only/kitti/${atk}/velodyne_purified"
    local meta="outputs/asap_hybrid_cluster_only/kitti/${atk}/meta.jsonl"
    local cfg="outputs/asap/configs/scorer_${atk}.json"
    local log="${LOGDIR}/asap_hybrid_cluster_only_${atk}.log"

    echo "[$(date +%H:%M:%S)] === ASAP hybrid (cluster-only) on ${atk} ==="
    "${PYTHON}" -m asap.pipeline \
        --input_dir "${in_dir}" \
        --out_dir "${out_dir}" \
        --meta_jsonl "${meta}" \
        --scorer_json "${cfg}" \
        --purifier hybrid \
        --hybrid_min_neighbors 0 \
        --hybrid_isolation_radius 0 \
        --seed 42 \
        --workers 16 \
        2>&1 | tee "${log}"
    echo "[$(date +%H:%M:%S)] === Done ${atk} ==="
}

run_one E2.1_injection
run_one E2.2_perturbation
run_one E2.3_dropping

echo "[$(date +%H:%M:%S)] All ASAP hybrid (cluster-only) runs complete."
