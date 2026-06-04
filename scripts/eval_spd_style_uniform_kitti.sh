#!/usr/bin/env bash
# Evaluate PointPillars on the SPD-style uniform score-net KITTI output.
set -euo pipefail

cd /root/autodl-tmp/ASAP

ATTACK="${ATTACK:-E2.2_perturbation}"
OUT_BASE="${OUT_BASE:-outputs/spd_style_uniform_score_net}"
TAG="${TAG:-E4_spd_style_uniform_${ATTACK}_pp}"
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
    --entry "SPD-style uniform ${ATTACK}:${TAG}"
