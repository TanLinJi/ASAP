#!/usr/bin/env bash
# Run no-defense nuScenes evals on keyframe-only E2.x attack sets.
#
# Default: VoxelNeXt (E4.3). Use DETECTOR=transfusion for E4.4.
#
# Recommended:
#   tmux new-session -d -s asap_nusc_voxelnext_eval \
#     "cd /root/autodl-tmp/ASAP && NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 DETECTOR=voxelnext bash scripts/eval_nuscenes_keyframe_no_defense.sh"

set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/ASAP}"
ATTACK_ROOT="${ATTACK_ROOT:-${ROOT}/outputs/attacks/nuscenes_keyframe}"
DETECTOR="${DETECTOR:-voxelnext}"
NUM_GPUS="${NUM_GPUS:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
DIST_PORT_BASE="${DIST_PORT_BASE:-19300}"

case "${DETECTOR}" in
    voxelnext)
        CFG="cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"
        CKPT="${ROOT}/checkpoints/nuscenes/voxelnext_nuscenes_kernel1.pth"
        MODEL="cbgs_voxel0075_voxelnext"
        TAG_PREFIX="E4_3_voxelnext_keyframe_no_defense"
        OUT_JSON="${ATTACK_ROOT}/nuscenes_voxelnext_no_defense_eval_summary.json"
        CLEAN_MAP="0.605195155366683"
        CLEAN_NDS="0.6664"
        ;;
    transfusion|transfusion_lidar)
        CFG="cfgs/nuscenes_models/transfusion_lidar.yaml"
        CKPT="${ROOT}/checkpoints/nuscenes/cbgs_transfusion_lidar.pth"
        MODEL="transfusion_lidar"
        TAG_PREFIX="E4_4_transfusion_lidar_keyframe_no_defense"
        OUT_JSON="${ATTACK_ROOT}/nuscenes_transfusion_lidar_no_defense_eval_summary.json"
        CLEAN_MAP="0.645704360834918"
        CLEAN_NDS="0.6943"
        ;;
    *)
        echo "ERROR: DETECTOR must be voxelnext or transfusion_lidar, got ${DETECTOR}" >&2
        exit 2
        ;;
esac

cd "${ROOT}"

has_metrics() {
    local tag="$1"
    find "third_party/OpenPCDet/output/nuscenes_models/${MODEL}/${tag}/eval" \
        -name 'metrics_summary.json' -size +0c -print -quit 2>/dev/null | grep -q .
}

run_one() {
    local attack_id="$1"
    local tag_suffix="$2"
    local port="$3"
    local tag="${TAG_PREFIX}_${tag_suffix}"
    local attack_dir="${ATTACK_ROOT}/${attack_id}"

    if [ ! -d "${attack_dir}/samples/LIDAR_TOP" ]; then
        echo "ERROR: missing attack set: ${attack_dir}/samples/LIDAR_TOP" >&2
        exit 1
    fi

    if has_metrics "${tag}"; then
        echo "[skip] ${tag} already has metrics_summary.json"
        return
    fi

    echo "[run] ${DETECTOR} ${attack_id} -> ${tag}"
    NUM_GPUS="${NUM_GPUS}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    EVAL_WORKERS="${EVAL_WORKERS}" \
    DIST_PORT="${port}" \
        bash scripts/eval_attacked_nuscenes.sh \
            "${attack_dir}" \
            "${CFG}" \
            "${CKPT}" \
            "${tag}"
}

run_one E2.1_injection E2_1 "$((DIST_PORT_BASE + 1))"
run_one E2.2_perturbation E2_2 "$((DIST_PORT_BASE + 2))"
run_one E2.3_dropping E2_3 "$((DIST_PORT_BASE + 3))"

/root/miniconda3/envs/asap/bin/python scripts/parse_nuscenes_eval_summary.py \
    --model "${MODEL}" \
    --clean-map "${CLEAN_MAP}" \
    --clean-nds "${CLEAN_NDS}" \
    --out "${OUT_JSON}" \
    --entry "E2.1 no-defense:${TAG_PREFIX}_E2_1" \
    --entry "E2.2 no-defense:${TAG_PREFIX}_E2_2" \
    --entry "E2.3 no-defense:${TAG_PREFIX}_E2_3"

echo "[done] wrote ${OUT_JSON}"
