#!/usr/bin/env bash
# Small threshold-sensitivity sweep for KITTI E2.2 Track A score-net.
#
# Runs selected tau overrides sequentially to avoid GPU contention, evaluates
# PointPillars after each purified set, and then parses a combined summary.
set -euo pipefail

cd /root/autodl-tmp/ASAP

CKPT="${CKPT:-checkpoints/kitti/asap_score_net_trackA_v1.pth}"
ATTACK="${ATTACK:-E2.2_perturbation}"
TAUS="${TAUS:-0.036 0.048}"
GPUS="${GPUS:-0,1}"
OUT_ROOT="${OUT_ROOT:-outputs/asap_tau_sensitivity}"
SCORE_ANCHOR_BLEND="${SCORE_ANCHOR_BLEND:-0.75}"

mkdir -p "${OUT_ROOT}/logs"

entries=()
for tau in ${TAUS}; do
    tau_tag="${tau/./p}"
    out_base="${OUT_ROOT}/tau_${tau_tag}"
    tag="E4_tau_${tau_tag}_${ATTACK}_pp"
    echo "[$(date +%H:%M:%S)] === tau sensitivity tau=${tau} out=${out_base} ==="

    OUT_BASE="${out_base}" \
    GPUS="${GPUS}" \
    SCORE_ANCHOR_BLEND="${SCORE_ANCHOR_BLEND}" \
    TAU_OVERRIDE="${tau}" \
        bash scripts/run_asap_vpsde_score_net_dual_gpu.sh "${CKPT}" "${ATTACK}" \
        2>&1 | tee "${OUT_ROOT}/logs/tau_${tau_tag}_purify.log"

    OUT_BASE="${out_base}" \
    TAG="${tag}" \
    ATTACK="${ATTACK}" \
    NUM_GPUS=2 \
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    EVAL_WORKERS=4 \
        bash scripts/eval_adaptive_all_spu_kitti.sh \
        2>&1 | tee "${OUT_ROOT}/logs/tau_${tau_tag}_eval.log"

    entries+=(--entry "tau=${tau} ${ATTACK}:${tag}")
done

python scripts/parse_eval_summary.py \
    --out "${OUT_ROOT}/kitti_pointpillars_tau_sensitivity_summary.json" \
    "${entries[@]}"
