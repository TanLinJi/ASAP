#!/usr/bin/env bash
# Run an SPD-style baseline: fixed-radius spherical units + all-unit score-net purification.
#
# This is not claimed as an official LiDAR-SPD reproduction. It is an
# internal external-idea baseline that removes ASAP's two key innovations:
# adaptive radius and anomaly-guided selective gating.
#
# Usage:
#   tmux new -s asap_spd_uniform_e22 \\
#     'ATTACK=E2.2_perturbation bash scripts/run_spd_style_uniform_score_net_dual_gpu.sh'
#
# Optional env:
#   ATTACK=E2.2_perturbation
#   GPUS=0,1
#   WORKERS_PER_GPU=1
#   CKPT=checkpoints/kitti/asap_score_net_trackA_v1.pth
#   OUT_BASE=outputs/spd_style_uniform_score_net
#   FIXED_RADIUS=0.40
#   INNER_BETA=0.67
#   STRIDE=0.40
#   MAX_CENTERS=
#   SCORE_ANCHOR_BLEND=0.75
#   SCORE_BATCH_SPUS=64
set -euo pipefail

cd /root/autodl-tmp/ASAP
export PYTHONPATH=src:${PYTHONPATH:-}

PYTHON=/root/miniconda3/envs/asap/bin/python
ATTACK="${ATTACK:-E2.2_perturbation}"
CKPT="${CKPT:-checkpoints/kitti/asap_score_net_trackA_v1.pth}"
OUT_BASE="${OUT_BASE:-outputs/spd_style_uniform_score_net}"
GPUS="${GPUS:-0,1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
FIXED_RADIUS="${FIXED_RADIUS:-0.40}"
INNER_BETA="${INNER_BETA:-0.67}"
STRIDE="${STRIDE:-0.40}"
MAX_CENTERS="${MAX_CENTERS:-}"
SCORE_ANCHOR_BLEND="${SCORE_ANCHOR_BLEND:-0.75}"
SCORE_BATCH_SPUS="${SCORE_BATCH_SPUS:-64}"
SCORE_STEP_SIZE="${SCORE_STEP_SIZE:-1.0}"
SCORE_CLIP="${SCORE_CLIP:-3.0}"
T_STAR="${T_STAR:-0.05}"

IN_DIR="outputs/attacks/kitti/${ATTACK}/velodyne_attacked"
CFG="outputs/asap/configs/scorer_${ATTACK}.json"
OUT_ROOT="${OUT_BASE}/kitti/${ATTACK}"
OUT_DIR="${OUT_ROOT}/velodyne_purified"
SHARD_ROOT="${OUT_ROOT}/shards"
LOGDIR="${OUT_BASE}/logs"
META="${OUT_ROOT}/meta.jsonl"
mkdir -p "${OUT_DIR}" "${SHARD_ROOT}" "${LOGDIR}"

if [ ! -d "${IN_DIR}" ]; then
    echo "ERROR: missing input dir ${IN_DIR}" >&2
    exit 1
fi
if [ ! -f "${CFG}" ]; then
    echo "ERROR: missing scorer config ${CFG}" >&2
    exit 1
fi
if [ ! -f "${CKPT}" ]; then
    echo "ERROR: missing checkpoint ${CKPT}" >&2
    exit 1
fi

rm -rf "${SHARD_ROOT}"
mkdir -p "${OUT_DIR}" "${SHARD_ROOT}"
rm -f "${META}"
find "${OUT_DIR}" -maxdepth 1 -name '*.bin' -delete

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
if [ "${#GPU_IDS[@]}" -lt 1 ]; then
    echo "ERROR: GPUS is empty" >&2
    exit 1
fi

for gpu in "${GPU_IDS[@]}"; do
    mkdir -p "${SHARD_ROOT}/input_gpu${gpu}" "${SHARD_ROOT}/output_gpu${gpu}"
done

i=0
while IFS= read -r f; do
    gpu="${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}"
    ln -sf "$(realpath "${f}")" "${SHARD_ROOT}/input_gpu${gpu}/$(basename "${f}")"
    i=$((i + 1))
done < <(find "${IN_DIR}" -maxdepth 1 -name '*.bin' | sort)

echo "[$(date +%H:%M:%S)] SPD-style uniform score-net baseline attack=${ATTACK} frames=${i} gpus=${GPUS}"
echo "[$(date +%H:%M:%S)] fixed_radius=${FIXED_RADIUS} stride=${STRIDE} beta=${INNER_BETA} max_centers=${MAX_CENTERS:-none}"

pids=()
for gpu in "${GPU_IDS[@]}"; do
    log="${LOGDIR}/spd_style_uniform_${ATTACK}_gpu${gpu}.log"
    max_args=()
    if [ -n "${MAX_CENTERS}" ]; then
        max_args=(--spu_max_centers "${MAX_CENTERS}")
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m asap.pipeline \
        --input_dir "${SHARD_ROOT}/input_gpu${gpu}" \
        --out_dir "${SHARD_ROOT}/output_gpu${gpu}" \
        --meta_jsonl "${SHARD_ROOT}/meta_gpu${gpu}.jsonl" \
        --scorer_json "${CFG}" \
        --purifier vp_sde \
        --tau -1.0 \
        --spu_alpha 0.0 \
        --spu_r_min "${FIXED_RADIUS}" \
        --spu_r_max "${FIXED_RADIUS}" \
        --spu_beta "${INNER_BETA}" \
        --spu_eta "${STRIDE}" \
        "${max_args[@]}" \
        --score_net_ckpt "${CKPT}" \
        --score_net_device cuda \
        --score_step_size "${SCORE_STEP_SIZE}" \
        --score_clip "${SCORE_CLIP}" \
        --score_batch_spus "${SCORE_BATCH_SPUS}" \
        --score_anchor_blend "${SCORE_ANCHOR_BLEND}" \
        --t_star "${T_STAR}" \
        --seed 42 \
        --workers "${WORKERS_PER_GPU}" \
        > "${log}" 2>&1 &
    pids+=("$!")
    echo "[$(date +%H:%M:%S)] launched gpu=${gpu} pid=${pids[-1]} log=${log}"
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
if [ "${failed}" -ne 0 ]; then
    echo "ERROR: at least one shard failed for ${ATTACK}" >&2
    exit 1
fi

for gpu in "${GPU_IDS[@]}"; do
    find "${SHARD_ROOT}/output_gpu${gpu}" -maxdepth 1 -name '*.bin' \
        -exec sh -c 'ln -sf "$(realpath "$1")" "$2/$(basename "$1")"' sh {} "${OUT_DIR}" \;
    cat "${SHARD_ROOT}/meta_gpu${gpu}.jsonl" >> "${META}"
done

produced="$(find "${OUT_DIR}" -maxdepth 1 -name '*.bin' | wc -l)"
meta_lines="$(wc -l < "${META}")"
echo "[$(date +%H:%M:%S)] done attack=${ATTACK}; produced=${produced}; meta_lines=${meta_lines}; out=${OUT_ROOT}"
