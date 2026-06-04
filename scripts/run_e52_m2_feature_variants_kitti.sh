#!/usr/bin/env bash
# E5.2 stage-2 detector validation for selected M2 feature variants.
#
# Runs KITTI E2.2 Perturbation only, because the lightweight diagnostic showed
# this is the decision-critical row: only_f_d nearly matches full M2 on the
# calibration set, while drop_f_d degrades the calibration signal.
set -euo pipefail

cd /root/autodl-tmp/ASAP

ATTACK="${ATTACK:-E2.2_perturbation}"
VARIANTS="${VARIANTS:-only_f_d drop_f_d}"
GPUS="${GPUS:-0,1}"
CKPT="${CKPT:-checkpoints/kitti/asap_score_net_trackA_v1.pth}"
OUT_ROOT="${OUT_ROOT:-outputs/ablations/E5.2_m2_features/stage2_detector}"
CONFIG_DIR="${CONFIG_DIR:-outputs/ablations/E5.2_m2_features/configs}"
SCORE_ANCHOR_BLEND="${SCORE_ANCHOR_BLEND:-0.75}"
SCORE_BATCH_SPUS="${SCORE_BATCH_SPUS:-64}"

mkdir -p "${OUT_ROOT}/logs"

entries=()
for variant in ${VARIANTS}; do
    scorer="${CONFIG_DIR}/scorer_perturbation_${variant}.json"
    if [ ! -f "${scorer}" ]; then
        echo "[$(date +%H:%M:%S)] scorer missing; exporting ${variant}"
        PYTHONPATH=src /root/miniconda3/envs/asap/bin/python \
            scripts/export_m2_feature_variant_scorers.py \
            --attack perturbation \
            --variant "${variant}" \
            --out_dir "${CONFIG_DIR}"
    fi

    out_base="${OUT_ROOT}/${variant}"
    tag="E5_2_${variant}_${ATTACK}_pp"
    echo "[$(date +%H:%M:%S)] === E5.2 ${variant}: purification -> ${out_base} ==="

    OUT_BASE="${out_base}" \
    GPUS="${GPUS}" \
    SCORE_ANCHOR_BLEND="${SCORE_ANCHOR_BLEND}" \
    SCORE_BATCH_SPUS="${SCORE_BATCH_SPUS}" \
    SCORER_JSON="${scorer}" \
        bash scripts/run_asap_vpsde_score_net_dual_gpu.sh "${CKPT}" "${ATTACK}" \
        2>&1 | tee "${OUT_ROOT}/logs/${variant}_purify.log"

    echo "[$(date +%H:%M:%S)] === E5.2 ${variant}: PointPillars eval tag=${tag} ==="
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
    --out "${OUT_ROOT}/kitti_pointpillars_e52_feature_variant_summary.json" \
    "${entries[@]}"

echo "[$(date +%H:%M:%S)] E5.2 stage-2 detector validation complete."
