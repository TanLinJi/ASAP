#!/usr/bin/env bash
# E5.1 detector validation for fixed-radius M1 variants.
#
# Runs KITTI PointPillars on E2.2 Perturbation with M2/M3 kept at the Track A
# settings and only M1 changed by forcing r_min = r_max = R.
set -euo pipefail

cd /root/autodl-tmp/ASAP

ATTACK="${ATTACK:-E2.2_perturbation}"
RADII="${RADII:-0.10 0.20 0.30 0.40}"
GPUS="${GPUS:-0,1}"
CKPT="${CKPT:-checkpoints/kitti/asap_score_net_trackA_v1.pth}"
OUT_ROOT="${OUT_ROOT:-outputs/ablations/E5.1_fixed_radius}"
SCORE_ANCHOR_BLEND="${SCORE_ANCHOR_BLEND:-0.75}"
SCORE_BATCH_SPUS="${SCORE_BATCH_SPUS:-64}"
SPU_ETA="${SPU_ETA:-0.30}"
SPU_ALPHA="${SPU_ALPHA:-0.0}"

mkdir -p "${OUT_ROOT}/logs"

entries=()
for radius in ${RADII}; do
    variant="fixed_r${radius//./}"
    out_base="${OUT_ROOT}/${variant}"
    tag="E5_1_${variant}_${ATTACK}_pp"

    echo "[$(date +%H:%M:%S)] === E5.1 ${variant}: purification -> ${out_base} ==="
    OUT_BASE="${out_base}" \
    GPUS="${GPUS}" \
    SCORE_ANCHOR_BLEND="${SCORE_ANCHOR_BLEND}" \
    SCORE_BATCH_SPUS="${SCORE_BATCH_SPUS}" \
    SPU_ALPHA="${SPU_ALPHA}" \
    SPU_R_MIN="${radius}" \
    SPU_R_MAX="${radius}" \
    SPU_ETA="${SPU_ETA}" \
        bash scripts/run_asap_vpsde_score_net_dual_gpu.sh "${CKPT}" "${ATTACK}" \
        2>&1 | tee "${OUT_ROOT}/logs/${variant}_purify.log"

    echo "[$(date +%H:%M:%S)] === E5.1 ${variant}: PointPillars eval tag=${tag} ==="
    OUT_BASE="${out_base}" \
    TAG="${tag}" \
    ATTACK="${ATTACK}" \
    NUM_GPUS=2 \
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    EVAL_WORKERS=4 \
        bash scripts/eval_adaptive_all_spu_kitti.sh \
        2>&1 | tee "${OUT_ROOT}/logs/${variant}_eval.log"

    entries+=(--entry "${variant}:${tag}")
done

python scripts/parse_eval_summary.py \
    --out "${OUT_ROOT}/kitti_pointpillars_e51_fixed_radius_summary.json" \
    "${entries[@]}"

echo "[$(date +%H:%M:%S)] E5.1 fixed-radius detector validation complete."
