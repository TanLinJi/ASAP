#!/usr/bin/env bash
# Evaluate PointPillars on all 3 ASAP-hybrid (cluster-only) purified KITTI val sets.
set -euo pipefail

cd /root/autodl-tmp/ASAP

CKPT=/root/autodl-tmp/ASAP/checkpoints/kitti/pointpillar_7728.pth
CFG=cfgs/kitti_models/pointpillar.yaml

LOGDIR=outputs/asap_hybrid_cluster_only/eval_logs
mkdir -p "${LOGDIR}"

for atk in E2.1_injection E2.2_perturbation E2.3_dropping; do
    tag="E4_asap_hybrid_co_$(echo ${atk} | sed 's/_.*//')_pp"
    log="${LOGDIR}/${tag}.log"
    echo "[$(date +%H:%M:%S)] === Eval ${tag} ==="
    bash scripts/eval_attacked_kitti.sh \
        outputs/asap_hybrid_cluster_only/kitti/${atk} \
        "${CFG}" \
        "${CKPT}" \
        "${tag}" \
        4 2>&1 | tee "${log}"
    echo "[$(date +%H:%M:%S)] === Done ${tag} ==="
done

echo "[$(date +%H:%M:%S)] All ASAP hybrid (cluster-only) evaluations complete."
