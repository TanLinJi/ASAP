#!/usr/bin/env bash
# Run ASAP VP-SDE score-net purification with two T4 GPUs via file sharding.
#
# Usage:
#   bash scripts/run_asap_vpsde_score_net_dual_gpu.sh <score_net_ckpt> [attack_id]
#
# Optional env:
#   GPUS=0,1
#   WORKERS_PER_GPU=1
#   T_STAR=0.05
#   SCORE_STEP_SIZE=1.0
#   SCORE_CLIP=3.0
#   SCORE_BATCH_SPUS=64
#   SCORE_ANCHOR_BLEND=0.0
#   TAU_OVERRIDE=
#   POLICY_MIN_VOTES=                  # optional purifier vote threshold
#   SCORE_INJECTION_FILTER_RADIUS=0.0
#   SCORE_INJECTION_FILTER_MIN_NEIGHBORS=0
#   SCORE_FILTER_ATTACK_KINDS=injection
#   SCORER_JSON=                         # override per-attack scorer config
#   SPU_ALPHA=2.0
#   SPU_BETA=0.67
#   SPU_R_MIN=0.10
#   SPU_R_MAX=0.40
#   SPU_ETA=0.30
#   SPU_MAX_CENTERS=
#   SPU_BACKEND=cpu
#   OUT_BASE=outputs/asap_vpsde_score_net
#
# To survive disconnects:
#   tmux new -s asap_score_e22 'bash scripts/run_asap_vpsde_score_net_dual_gpu.sh checkpoints/kitti/asap_score_net_trackA_v1.pth E2.2_perturbation'
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/run_asap_vpsde_score_net_dual_gpu.sh <score_net_ckpt> [attack_id]" >&2
    exit 1
fi

cd /root/autodl-tmp/ASAP
export PYTHONPATH=src:${PYTHONPATH:-}

CKPT="$1"
ONLY_ATTACK="${2:-}"
PYTHON=/root/miniconda3/envs/asap/bin/python
OUT_BASE="${OUT_BASE:-outputs/asap_vpsde_score_net}"
LOGDIR="${OUT_BASE}/logs"
GPUS="${GPUS:-0,1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
T_STAR="${T_STAR:-0.05}"
SCORE_STEP_SIZE="${SCORE_STEP_SIZE:-1.0}"
SCORE_CLIP="${SCORE_CLIP:-3.0}"
SCORE_BATCH_SPUS="${SCORE_BATCH_SPUS:-64}"
SCORE_ANCHOR_BLEND="${SCORE_ANCHOR_BLEND:-0.0}"
TAU_OVERRIDE="${TAU_OVERRIDE:-}"
POLICY_MIN_VOTES="${POLICY_MIN_VOTES:-}"
SCORER_JSON="${SCORER_JSON:-}"
SCORE_INJECTION_FILTER_RADIUS="${SCORE_INJECTION_FILTER_RADIUS:-0.0}"
SCORE_INJECTION_FILTER_MIN_NEIGHBORS="${SCORE_INJECTION_FILTER_MIN_NEIGHBORS:-0}"
SCORE_FILTER_ATTACK_KINDS="${SCORE_FILTER_ATTACK_KINDS:-injection}"
SPU_ALPHA="${SPU_ALPHA:-2.0}"
SPU_BETA="${SPU_BETA:-0.67}"
SPU_R_MIN="${SPU_R_MIN:-0.10}"
SPU_R_MAX="${SPU_R_MAX:-0.40}"
SPU_ETA="${SPU_ETA:-0.30}"
SPU_MAX_CENTERS="${SPU_MAX_CENTERS:-}"
SPU_BACKEND="${SPU_BACKEND:-cpu}"
mkdir -p "${LOGDIR}"

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
if [ "${#GPU_IDS[@]}" -lt 1 ]; then
    echo "ERROR: GPUS is empty" >&2
    exit 1
fi

run_one () {
    local atk="$1"
    local in_dir="outputs/attacks/kitti/${atk}/velodyne_attacked"
    local out_root="${OUT_BASE}/kitti/${atk}"
    local out_dir="${out_root}/velodyne_purified"
    local meta="${out_root}/meta.jsonl"
    local shard_root="${out_root}/shards"
    local cfg="${SCORER_JSON:-outputs/asap/configs/scorer_${atk}.json}"
    if [ ! -f "${cfg}" ]; then
        echo "ERROR: scorer config not found: ${cfg}" >&2
        exit 1
    fi

    rm -rf "${shard_root}"
    mkdir -p "${out_dir}" "${shard_root}"
    rm -f "${meta}"
    find "${out_dir}" -maxdepth 1 -name '*.bin' -delete

    local n_gpu="${#GPU_IDS[@]}"
    local i=0
    for gpu in "${GPU_IDS[@]}"; do
        mkdir -p "${shard_root}/input_gpu${gpu}" "${shard_root}/output_gpu${gpu}"
    done

    while IFS= read -r f; do
        local gpu="${GPU_IDS[$((i % n_gpu))]}"
        ln -sf "$(realpath "${f}")" "${shard_root}/input_gpu${gpu}/$(basename "${f}")"
        i=$((i + 1))
    done < <(find "${in_dir}" -maxdepth 1 -name '*.bin' | sort)

    echo "[$(date +%H:%M:%S)] === ASAP vp_sde score-net dual-GPU on ${atk}; frames=${i}; gpus=${GPUS} ==="
    if [ -n "${TAU_OVERRIDE}" ]; then
        echo "[$(date +%H:%M:%S)] tau override enabled: ${TAU_OVERRIDE}"
    fi
    echo "[$(date +%H:%M:%S)] scorer config: ${cfg}"
    local pids=()
    for gpu in "${GPU_IDS[@]}"; do
        local log="${LOGDIR}/asap_vpsde_score_net_${atk}_gpu${gpu}.log"
        local tau_args=()
        if [ -n "${TAU_OVERRIDE}" ]; then
            tau_args=(--tau "${TAU_OVERRIDE}")
        fi
        local vote_args=()
        if [ -n "${POLICY_MIN_VOTES}" ]; then
            vote_args=(--policy_min_votes "${POLICY_MIN_VOTES}")
        fi
        local max_center_args=()
        if [ -n "${SPU_MAX_CENTERS}" ]; then
            max_center_args=(--spu_max_centers "${SPU_MAX_CENTERS}")
        fi
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m asap.pipeline \
            --input_dir "${shard_root}/input_gpu${gpu}" \
            --out_dir "${shard_root}/output_gpu${gpu}" \
            --meta_jsonl "${shard_root}/meta_gpu${gpu}.jsonl" \
            --scorer_json "${cfg}" \
            --purifier vp_sde \
            "${tau_args[@]}" \
            "${vote_args[@]}" \
            --spu_alpha "${SPU_ALPHA}" \
            --spu_beta "${SPU_BETA}" \
            --spu_r_min "${SPU_R_MIN}" \
            --spu_r_max "${SPU_R_MAX}" \
            --spu_eta "${SPU_ETA}" \
            --spu_backend "${SPU_BACKEND}" \
            "${max_center_args[@]}" \
            --score_net_ckpt "${CKPT}" \
            --score_net_device cuda \
            --score_step_size "${SCORE_STEP_SIZE}" \
            --score_clip "${SCORE_CLIP}" \
            --score_batch_spus "${SCORE_BATCH_SPUS}" \
            --score_anchor_blend "${SCORE_ANCHOR_BLEND}" \
            --score_injection_filter_radius "${SCORE_INJECTION_FILTER_RADIUS}" \
            --score_injection_filter_min_neighbors "${SCORE_INJECTION_FILTER_MIN_NEIGHBORS}" \
            --score_filter_attack_kinds "${SCORE_FILTER_ATTACK_KINDS}" \
            --t_star "${T_STAR}" \
            --seed 42 \
            --workers "${WORKERS_PER_GPU}" \
            > "${log}" 2>&1 &
        pids+=("$!")
        echo "[$(date +%H:%M:%S)] launched ${atk} gpu=${gpu} pid=${pids[-1]} log=${log}"
    done

    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if [ "${failed}" -ne 0 ]; then
        echo "ERROR: at least one shard failed for ${atk}" >&2
        exit 1
    fi

    for gpu in "${GPU_IDS[@]}"; do
        find "${shard_root}/output_gpu${gpu}" -maxdepth 1 -name '*.bin' -exec ln -sf "$(pwd)/{}" "${out_dir}/" \;
        cat "${shard_root}/meta_gpu${gpu}.jsonl" >> "${meta}"
    done

    local produced
    produced="$(find "${out_dir}" -maxdepth 1 -name '*.bin' | wc -l)"
    local meta_lines
    meta_lines="$(wc -l < "${meta}")"
    echo "[$(date +%H:%M:%S)] === Done ${atk}; produced=${produced}; meta_lines=${meta_lines} ==="
}

if [ -n "${ONLY_ATTACK}" ]; then
    run_one "${ONLY_ATTACK}"
else
    run_one E2.1_injection
    run_one E2.2_perturbation
    run_one E2.3_dropping
fi

echo "[$(date +%H:%M:%S)] ASAP vp_sde score-net dual-GPU runs complete."
