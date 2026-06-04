#!/usr/bin/env bash
# Evaluate PV-RCNN on KITTI SOR/ROR defended attack sets.
#
# Runs six sequential cells:
#   E3.1 SOR x E2.1/E2.2/E2.3
#   E3.2 ROR x E2.1/E2.2/E2.3
#
# Use from tmux for long runs:
#   NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash scripts/eval_pvrcnn_sor_ror_kitti.sh

set -euo pipefail

ROOT="/root/autodl-tmp/ASAP"
CFG="cfgs/kitti_models/pv_rcnn.yaml"
CKPT="${ROOT}/checkpoints/kitti/pv_rcnn_8369.pth"
OUT_JSON="${ROOT}/outputs/asap_vpsde_score_net_trackA_revised_dropping_skip/kitti_pvrcnn_sor_ror_eval_summary.json"

BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_GPUS="${NUM_GPUS:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
DIST_PORT_BASE="${DIST_PORT_BASE:-19110}"

cd "${ROOT}"

has_nonempty_eval_log() {
    local tag="$1"
    find "third_party/OpenPCDet/output/kitti_models/pv_rcnn/${tag}/eval" \
        -name 'log_eval_*.txt' -size +0c -print -quit 2>/dev/null | grep -q .
}

run_one() {
    local attack_dir="$1"
    local tag="$2"
    local port="$3"

    if has_nonempty_eval_log "${tag}"; then
        echo "[skip] ${tag} already has a non-empty eval log"
        return
    fi

    echo "[run] ${tag}"
    NUM_GPUS="${NUM_GPUS}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    EVAL_WORKERS="${EVAL_WORKERS}" \
    DIST_PORT="${port}" \
        bash scripts/eval_attacked_kitti.sh \
            "${attack_dir}" \
            "${CFG}" \
            "${CKPT}" \
            "${tag}" \
            "${BATCH_SIZE}"
}

run_one "outputs/defenses/kitti/E3.1_sor/E2.1_injection" \
    "E4_2_pvrcnn_sor_E2_1" "$((DIST_PORT_BASE + 1))"
run_one "outputs/defenses/kitti/E3.1_sor/E2.2_perturbation" \
    "E4_2_pvrcnn_sor_E2_2" "$((DIST_PORT_BASE + 2))"
run_one "outputs/defenses/kitti/E3.1_sor/E2.3_dropping" \
    "E4_2_pvrcnn_sor_E2_3" "$((DIST_PORT_BASE + 3))"
run_one "outputs/defenses/kitti/E3.2_ror/E2.1_injection" \
    "E4_2_pvrcnn_ror_E2_1" "$((DIST_PORT_BASE + 4))"
run_one "outputs/defenses/kitti/E3.2_ror/E2.2_perturbation" \
    "E4_2_pvrcnn_ror_E2_2" "$((DIST_PORT_BASE + 5))"
run_one "outputs/defenses/kitti/E3.2_ror/E2.3_dropping" \
    "E4_2_pvrcnn_ror_E2_3" "$((DIST_PORT_BASE + 6))"

/root/miniconda3/envs/asap/bin/python scripts/parse_eval_summary.py \
    --model pv_rcnn \
    --out "${OUT_JSON}" \
    --entry "E2.1 SOR:E4_2_pvrcnn_sor_E2_1" \
    --entry "E2.2 SOR:E4_2_pvrcnn_sor_E2_2" \
    --entry "E2.3 SOR:E4_2_pvrcnn_sor_E2_3" \
    --entry "E2.1 ROR:E4_2_pvrcnn_ror_E2_1" \
    --entry "E2.2 ROR:E4_2_pvrcnn_ror_E2_2" \
    --entry "E2.3 ROR:E4_2_pvrcnn_ror_E2_3"

echo "[done] wrote ${OUT_JSON}"
