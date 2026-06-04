#!/usr/bin/env bash
# Evaluate PointPillars on adaptive-radius all-SPU score-net KITTI outputs.
set -euo pipefail

cd /root/autodl-tmp/ASAP

ATTACK="${ATTACK:-E2.2_perturbation}"
OUT_BASE="${OUT_BASE:-outputs/adaptive_all_spu_score_net}"
TAG="${TAG:-E4_adaptive_all_spu_${ATTACK}_pp}"
CKPT="${CKPT:-/root/autodl-tmp/ASAP/checkpoints/kitti/pointpillar_7728.pth}"
CFG="${CFG:-cfgs/kitti_models/pointpillar.yaml}"
NUM_FEATURES="${NUM_FEATURES:-4}"

LOGDIR="${OUT_BASE}/eval_logs"
mkdir -p "${LOGDIR}"

bash scripts/eval_attacked_kitti.sh \
    "${OUT_BASE}/kitti/${ATTACK}" \
    "${CFG}" \
    "${CKPT}" \
    "${TAG}" \
    "${NUM_FEATURES}" \
    2>&1 | tee "${LOGDIR}/${TAG}.log"

python scripts/parse_eval_summary.py \
    --out "${OUT_BASE}/kitti_pointpillars_eval_summary.json" \
    --entry "Adaptive all-SPU ${ATTACK}:${TAG}"
