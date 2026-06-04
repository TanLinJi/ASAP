#!/usr/bin/env bash
# Generate and evaluate defenses for the accepted nuScenes keyframe-only attacks.
#
# Accepted attacks:
#   - E2.1 Injection, original keyframe-only budget
#   - E2.2 Perturbation, budget-gated epsilon=0.5
#
# Dropping is intentionally excluded because keyframe-only drop_ratio=1.0 still
# has weak VoxelNeXt ASR under MAX_SWEEPS=10.
#
# Default run:
#   NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 DETECTOR=all DEFENSES=sor,ror \
#       bash scripts/eval_nuscenes_accepted_defenses.sh
#
# Disconnect-safe run:
#   tmux new -s asap_nusc_defenses_$(date +%Y%m%d_%H%M%S) \
#       'cd /root/autodl-tmp/ASAP && NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 DETECTOR=all DEFENSES=sor,ror bash scripts/eval_nuscenes_accepted_defenses.sh'
#
# ASAP is supported only when explicit nuScenes Track-A assets are provided:
#   DEFENSES=asap ASAP_SCORER_JSON_DIR=... ASAP_SCORE_NET_CKPT=...

set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/ASAP}"
ATTACK_ROOT="${ATTACK_ROOT:-${ROOT}/outputs/attacks/nuscenes_keyframe}"
BUDGET_ROOT="${BUDGET_ROOT:-${ROOT}/outputs/attacks/nuscenes_keyframe_budget_gate}"
DEFENSE_ROOT="${DEFENSE_ROOT:-${ROOT}/outputs/defenses/nuscenes_keyframe_accepted}"
DETECTOR="${DETECTOR:-all}"
DEFENSES="${DEFENSES:-sor,ror}"
ATTACKS="${ATTACKS:-E2_1,E2_2_eps0p5}"
NUM_GPUS="${NUM_GPUS:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
DIST_PORT_BASE="${DIST_PORT_BASE:-19600}"
TAG_SUFFIX="${TAG_SUFFIX:-}"
PYTHON="${PYTHON:-/root/miniconda3/envs/asap/bin/python}"
FORCE_PURIFY="${FORCE_PURIFY:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

SOR_K_NEIGHBORS="${SOR_K_NEIGHBORS:-20}"
SOR_ALPHA="${SOR_ALPHA:-1.0}"
ROR_RADIUS="${ROR_RADIUS:-0.4}"
ROR_MIN_NEIGHBORS="${ROR_MIN_NEIGHBORS:-5}"

ASAP_SCORER_JSON_DIR="${ASAP_SCORER_JSON_DIR:-}"
ASAP_SCORE_NET_CKPT="${ASAP_SCORE_NET_CKPT:-}"
ASAP_SCORE_ANCHOR_BLEND="${ASAP_SCORE_ANCHOR_BLEND:-0.75}"
ASAP_SCORE_INJECTION_FILTER_RADIUS="${ASAP_SCORE_INJECTION_FILTER_RADIUS:-0.4}"
ASAP_SCORE_INJECTION_FILTER_MIN_NEIGHBORS="${ASAP_SCORE_INJECTION_FILTER_MIN_NEIGHBORS:-5}"
ASAP_SCORE_FILTER_ATTACK_KINDS="${ASAP_SCORE_FILTER_ATTACK_KINDS:-injection}"
ASAP_T_STAR="${ASAP_T_STAR:-0.05}"
ASAP_SCORE_STEP_SIZE="${ASAP_SCORE_STEP_SIZE:-1.0}"
ASAP_SCORE_CLIP="${ASAP_SCORE_CLIP:-3.0}"
ASAP_GPUS="${ASAP_GPUS:-${CUDA_VISIBLE_DEVICES}}"
ASAP_WORKERS_PER_GPU="${ASAP_WORKERS_PER_GPU:-1}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

mkdir -p "${DEFENSE_ROOT}/logs"
cd "${ROOT}"

IFS=',' read -r -a DEFENSE_LIST <<< "${DEFENSES}"
IFS=',' read -r -a ATTACK_LIST <<< "${ATTACKS}"

case "${DETECTOR}" in
    all)
        DETECTOR_LIST=(voxelnext transfusion_lidar)
        ;;
    voxelnext)
        DETECTOR_LIST=(voxelnext)
        ;;
    transfusion|transfusion_lidar)
        DETECTOR_LIST=(transfusion_lidar)
        ;;
    *)
        echo "ERROR: DETECTOR must be all, voxelnext, or transfusion_lidar; got ${DETECTOR}" >&2
        exit 2
        ;;
esac

set_attack() {
    case "$1" in
        E2_1)
            ATTACK_LABEL="E2.1 Injection"
            ATTACK_PARSE_LABEL="E2.1"
            ATTACK_KIND="injection"
            ATTACK_SUFFIX="E2_1"
            ATTACK_NAME="E2.1_injection"
            ATTACK_DIR="${ATTACK_ROOT}/E2.1_injection"
            ;;
        E2_2_eps0p5)
            ATTACK_LABEL="E2.2 Perturbation eps0.5"
            ATTACK_PARSE_LABEL="E2.2 eps0.5"
            ATTACK_KIND="perturbation"
            ATTACK_SUFFIX="E2_2_eps0p5"
            ATTACK_NAME="E2.2_perturbation_eps0p5"
            ATTACK_DIR="${BUDGET_ROOT}/E2.2_perturbation_eps0p5"
            ;;
        *)
            echo "ERROR: unknown attack key $1" >&2
            exit 2
            ;;
    esac
}

set_detector() {
    case "$1" in
        voxelnext)
            CFG="cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"
            CKPT="${ROOT}/checkpoints/nuscenes/voxelnext_nuscenes_kernel1.pth"
            MODEL="cbgs_voxel0075_voxelnext"
            TAG_PREFIX="E4_3_voxelnext_accepted"
            OUT_JSON="${DEFENSE_ROOT}/nuscenes_voxelnext_accepted_defenses_eval_summary.json"
            CLEAN_MAP="0.605195155366683"
            CLEAN_NDS="0.6664"
            ;;
        transfusion_lidar)
            CFG="cfgs/nuscenes_models/transfusion_lidar.yaml"
            CKPT="${ROOT}/checkpoints/nuscenes/cbgs_transfusion_lidar.pth"
            MODEL="transfusion_lidar"
            TAG_PREFIX="E4_4_transfusion_lidar_accepted"
            OUT_JSON="${DEFENSE_ROOT}/nuscenes_transfusion_lidar_accepted_defenses_eval_summary.json"
            CLEAN_MAP="0.645704360834918"
            CLEAN_NDS="0.6943"
            ;;
        *)
            echo "ERROR: unknown detector $1" >&2
            exit 2
            ;;
    esac
}

count_bins() {
    if [ -d "$1" ]; then
        find "$1" -maxdepth 1 -name '*.bin' 2>/dev/null | wc -l
    else
        echo 0
    fi
}

has_metrics() {
    local model="$1"
    local tag="$2"
    find "third_party/OpenPCDet/output/nuscenes_models/${model}/${tag}/eval" \
        -name 'metrics_summary.json' -size +0c -print -quit 2>/dev/null | grep -q .
}

defense_dir() {
    local defense="$1"
    local attack_name="$2"
    echo "${DEFENSE_ROOT}/${defense}/${attack_name}"
}

generate_sor_ror() {
    local defense="$1"
    local attack_dir="$2"
    local attack_name="$3"
    local input_lidar="${attack_dir}/samples/LIDAR_TOP"
    local out_root
    out_root="$(defense_dir "${defense}" "${attack_name}")"
    local out_lidar="${out_root}/samples/LIDAR_TOP"
    local expected
    local produced
    expected="$(count_bins "${input_lidar}")"
    produced="$(count_bins "${out_lidar}")"

    if [ "${expected}" -eq 0 ]; then
        echo "ERROR: missing attack bins under ${input_lidar}" >&2
        exit 1
    fi
    if [ "${FORCE_PURIFY}" != "1" ] && [ "${produced}" -eq "${expected}" ]; then
        echo "[skip] ${defense} ${attack_name}: ${produced}/${expected} defended files already exist"
        return
    fi

    mkdir -p "${out_lidar}"
    find "${out_lidar}" -maxdepth 1 -name '*.bin' -delete
    local log="${DEFENSE_ROOT}/logs/${defense}_${attack_name}.log"
    echo "[run] ${defense} ${attack_name}: ${input_lidar} -> ${out_lidar}"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[dry-run] PYTHONPATH=${PYTHONPATH} ${PYTHON} -m asap.defenses.statistical_filters --method ${defense} --input_dir ${input_lidar} --out_dir ${out_lidar} --num_features 5"
        return
    fi

    if [ "${defense}" = "sor" ]; then
        "${PYTHON}" -m asap.defenses.statistical_filters \
            --method sor \
            --input_dir "${input_lidar}" \
            --out_dir "${out_lidar}" \
            --k_neighbors "${SOR_K_NEIGHBORS}" \
            --alpha "${SOR_ALPHA}" \
            --num_features 5 \
            2>&1 | tee "${log}"
    elif [ "${defense}" = "ror" ]; then
        "${PYTHON}" -m asap.defenses.statistical_filters \
            --method ror \
            --input_dir "${input_lidar}" \
            --out_dir "${out_lidar}" \
            --radius "${ROR_RADIUS}" \
            --min_neighbors "${ROR_MIN_NEIGHBORS}" \
            --num_features 5 \
            2>&1 | tee "${log}"
    else
        echo "ERROR: unsupported statistical defense ${defense}" >&2
        exit 2
    fi
}

generate_asap() {
    local attack_dir="$1"
    local attack_name="$2"
    local attack_suffix="$3"
    local attack_kind="$4"

    if [ -z "${ASAP_SCORE_NET_CKPT}" ] || [ ! -f "${ASAP_SCORE_NET_CKPT}" ]; then
        echo "ERROR: DEFENSES=asap requires ASAP_SCORE_NET_CKPT to point to a nuScenes Track-A score-net checkpoint." >&2
        exit 2
    fi
    if [ -z "${ASAP_SCORER_JSON_DIR}" ] || [ ! -d "${ASAP_SCORER_JSON_DIR}" ]; then
        echo "ERROR: DEFENSES=asap requires ASAP_SCORER_JSON_DIR with per-attack nuScenes scorer configs." >&2
        exit 2
    fi

    local scorer="${ASAP_SCORER_JSON_DIR}/scorer_${attack_suffix}.json"
    if [ ! -f "${scorer}" ]; then
        echo "ERROR: missing ASAP scorer config ${scorer}" >&2
        exit 2
    fi

    local input_lidar="${attack_dir}/samples/LIDAR_TOP"
    local out_root
    out_root="$(defense_dir "asap" "${attack_name}")"
    local out_lidar="${out_root}/samples/LIDAR_TOP"
    local meta="${out_root}/meta.jsonl"
    local shard_root="${out_root}/shards"
    local expected
    local produced
    expected="$(count_bins "${input_lidar}")"
    produced="$(count_bins "${out_lidar}")"

    if [ "${expected}" -eq 0 ]; then
        echo "ERROR: missing attack bins under ${input_lidar}" >&2
        exit 1
    fi
    if [ "${FORCE_PURIFY}" != "1" ] && [ "${produced}" -eq "${expected}" ] && [ -s "${meta}" ]; then
        echo "[skip] asap ${attack_name}: ${produced}/${expected} purified files already exist"
        return
    fi

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[dry-run] shard ASAP ${attack_name} over GPUS=${ASAP_GPUS}; scorer=${scorer}; ckpt=${ASAP_SCORE_NET_CKPT}"
        return
    fi

    rm -rf "${shard_root}"
    mkdir -p "${out_lidar}" "${shard_root}"
    find "${out_lidar}" -maxdepth 1 -name '*.bin' -delete
    rm -f "${meta}"

    IFS=',' read -r -a GPU_IDS <<< "${ASAP_GPUS}"
    if [ "${#GPU_IDS[@]}" -lt 1 ]; then
        echo "ERROR: ASAP_GPUS is empty" >&2
        exit 2
    fi

    local n_gpu="${#GPU_IDS[@]}"
    local i=0
    local gpu
    for gpu in "${GPU_IDS[@]}"; do
        mkdir -p "${shard_root}/input_gpu${gpu}" "${shard_root}/output_gpu${gpu}"
    done

    while IFS= read -r f; do
        gpu="${GPU_IDS[$((i % n_gpu))]}"
        ln -sf "$(realpath "${f}")" "${shard_root}/input_gpu${gpu}/$(basename "${f}")"
        i=$((i + 1))
    done < <(find "${input_lidar}" -maxdepth 1 -name '*.bin' | sort)

    echo "[run] asap ${attack_name}: frames=${i}, gpus=${ASAP_GPUS}, out=${out_lidar}"
    local pids=()
    for gpu in "${GPU_IDS[@]}"; do
        local log="${DEFENSE_ROOT}/logs/asap_${attack_name}_gpu${gpu}.log"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m asap.pipeline \
            --input_dir "${shard_root}/input_gpu${gpu}" \
            --out_dir "${shard_root}/output_gpu${gpu}" \
            --meta_jsonl "${shard_root}/meta_gpu${gpu}.jsonl" \
            --scorer_json "${scorer}" \
            --purifier vp_sde \
            --score_net_ckpt "${ASAP_SCORE_NET_CKPT}" \
            --score_net_device cuda \
            --score_step_size "${ASAP_SCORE_STEP_SIZE}" \
            --score_clip "${ASAP_SCORE_CLIP}" \
            --score_anchor_blend "${ASAP_SCORE_ANCHOR_BLEND}" \
            --score_injection_filter_radius "${ASAP_SCORE_INJECTION_FILTER_RADIUS}" \
            --score_injection_filter_min_neighbors "${ASAP_SCORE_INJECTION_FILTER_MIN_NEIGHBORS}" \
            --score_filter_attack_kinds "${ASAP_SCORE_FILTER_ATTACK_KINDS}" \
            --t_star "${ASAP_T_STAR}" \
            --attack_kind "${attack_kind}" \
            --seed 42 \
            --num_features 5 \
            --workers "${ASAP_WORKERS_PER_GPU}" \
            > "${log}" 2>&1 &
        pids+=("$!")
        echo "[launch] asap ${attack_name} gpu=${gpu} pid=${pids[-1]} log=${log}"
    done

    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if [ "${failed}" -ne 0 ]; then
        echo "ERROR: at least one ASAP shard failed for ${attack_name}" >&2
        exit 1
    fi

    for gpu in "${GPU_IDS[@]}"; do
        find "${shard_root}/output_gpu${gpu}" -maxdepth 1 -name '*.bin' \
            -exec sh -c 'ln -sf "$(realpath "$1")" "$2/$(basename "$1")"' sh {} "${out_lidar}" \;
        cat "${shard_root}/meta_gpu${gpu}.jsonl" >> "${meta}"
    done

    produced="$(count_bins "${out_lidar}")"
    echo "[done] asap ${attack_name}: produced=${produced}/${expected}, meta=$(wc -l < "${meta}")"
}

generate_defense() {
    local defense="$1"
    local attack_key="$2"
    set_attack "${attack_key}"
    case "${defense}" in
        sor|ror)
            generate_sor_ror "${defense}" "${ATTACK_DIR}" "${ATTACK_NAME}"
            ;;
        asap)
            generate_asap "${ATTACK_DIR}" "${ATTACK_NAME}" "${ATTACK_SUFFIX}" "${ATTACK_KIND}"
            ;;
        *)
            echo "ERROR: DEFENSES entries must be sor, ror, or asap; got ${defense}" >&2
            exit 2
            ;;
    esac
}

eval_one() {
    local detector="$1"
    local defense="$2"
    local attack_key="$3"
    set_detector "${detector}"
    set_attack "${attack_key}"

    local defended
    defended="$(defense_dir "${defense}" "${ATTACK_NAME}")"
    local tag="${TAG_PREFIX}_${defense}_${ATTACK_SUFFIX}${TAG_SUFFIX}"
    local port_offset
    case "${detector}_${defense}_${attack_key}" in
        voxelnext_sor_E2_1) port_offset=1 ;;
        voxelnext_sor_E2_2_eps0p5) port_offset=2 ;;
        voxelnext_ror_E2_1) port_offset=3 ;;
        voxelnext_ror_E2_2_eps0p5) port_offset=4 ;;
        voxelnext_asap_E2_1) port_offset=5 ;;
        voxelnext_asap_E2_2_eps0p5) port_offset=6 ;;
        transfusion_lidar_sor_E2_1) port_offset=11 ;;
        transfusion_lidar_sor_E2_2_eps0p5) port_offset=12 ;;
        transfusion_lidar_ror_E2_1) port_offset=13 ;;
        transfusion_lidar_ror_E2_2_eps0p5) port_offset=14 ;;
        transfusion_lidar_asap_E2_1) port_offset=15 ;;
        transfusion_lidar_asap_E2_2_eps0p5) port_offset=16 ;;
        *) port_offset=30 ;;
    esac

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[dry-run] NUM_GPUS=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} EVAL_WORKERS=${EVAL_WORKERS} DIST_PORT=$((DIST_PORT_BASE + port_offset)) bash scripts/eval_attacked_nuscenes.sh ${defended} ${CFG} ${CKPT} ${tag}"
        return
    fi

    if [ ! -d "${defended}/samples/LIDAR_TOP" ]; then
        echo "ERROR: defended set is missing: ${defended}/samples/LIDAR_TOP" >&2
        exit 1
    fi

    if [ "${FORCE_EVAL}" != "1" ] && has_metrics "${MODEL}" "${tag}"; then
        echo "[skip] ${detector} ${defense} ${ATTACK_LABEL}: ${tag} already has metrics_summary.json"
        return
    fi

    echo "[eval] ${detector} ${defense} ${ATTACK_LABEL} -> ${tag}"
    NUM_GPUS="${NUM_GPUS}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    EVAL_WORKERS="${EVAL_WORKERS}" \
    DIST_PORT="$((DIST_PORT_BASE + port_offset))" \
        bash scripts/eval_attacked_nuscenes.sh \
            "${defended}" \
            "${CFG}" \
            "${CKPT}" \
            "${tag}"
}

parse_detector() {
    local detector="$1"
    set_detector "${detector}"
    local args=(
        "${PYTHON}" scripts/parse_nuscenes_eval_summary.py
        --model "${MODEL}"
        --clean-map "${CLEAN_MAP}"
        --clean-nds "${CLEAN_NDS}"
        --out "${OUT_JSON}"
        --merge-existing
    )
    local defense
    for defense in "${DEFENSE_LIST[@]}"; do
        local attack_key
        for attack_key in "${ATTACK_LIST[@]}"; do
            set_attack "${attack_key}"
            args+=(--entry "${ATTACK_PARSE_LABEL} ${defense}:${TAG_PREFIX}_${defense}_${ATTACK_SUFFIX}${TAG_SUFFIX}")
        done
    done

    echo "[parse] ${detector} -> ${OUT_JSON}"
    "${args[@]}"
}

for defense in "${DEFENSE_LIST[@]}"; do
    for attack_key in "${ATTACK_LIST[@]}"; do
        generate_defense "${defense}" "${attack_key}"
    done
done

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] generation complete; eval commands:"
fi

for detector in "${DETECTOR_LIST[@]}"; do
    for defense in "${DEFENSE_LIST[@]}"; do
        for attack_key in "${ATTACK_LIST[@]}"; do
            eval_one "${detector}" "${defense}" "${attack_key}"
        done
    done
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[dry-run] skip parser for ${detector}"
    else
        parse_detector "${detector}"
    fi
done

echo "[done] nuScenes accepted defenses complete: ${DEFENSES} / ${DETECTOR}"
