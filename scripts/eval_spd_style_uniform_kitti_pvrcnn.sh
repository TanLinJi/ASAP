#!/usr/bin/env bash
# Evaluate PV-RCNN on the SPD-style uniform score-net KITTI output.
set -euo pipefail

cd /root/autodl-tmp/ASAP

ATTACK="${ATTACK:-E2.2_perturbation}"
OUT_BASE="${OUT_BASE:-outputs/spd_style_uniform_score_net_r015}"
TAG="${TAG:-E_sota_spd_r015_${ATTACK}_pvrcnn}"
CKPT="${CKPT:-/root/autodl-tmp/ASAP/checkpoints/kitti/pv_rcnn_8369.pth}"
CFG="${CFG:-cfgs/kitti_models/pv_rcnn.yaml}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DIST_PORT="${DIST_PORT:-19240}"

LOGDIR="${OUT_BASE}/eval_logs"
mkdir -p "${LOGDIR}"

NUM_GPUS="${NUM_GPUS:-2}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
EVAL_WORKERS="${EVAL_WORKERS:-4}" \
DIST_PORT="${DIST_PORT}" \
bash scripts/eval_attacked_kitti.sh \
    "${OUT_BASE}/kitti/${ATTACK}" \
    "${CFG}" \
    "${CKPT}" \
    "${TAG}" \
    "${BATCH_SIZE}" \
    2>&1 | tee "${LOGDIR}/${TAG}.log"

/root/miniconda3/envs/asap/bin/python scripts/parse_eval_summary.py \
    --model pv_rcnn \
    --out "${OUT_BASE}/kitti_pvrcnn_eval_summary.json" \
    --entry "SPD-style uniform ${ATTACK}:${TAG}"
