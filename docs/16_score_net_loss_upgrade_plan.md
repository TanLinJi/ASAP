# Score-net Loss Upgrade Experiment Plan — 2026-06-04

## Motivation

The current score-net is trained with a basic VP-SDE denoising score matching objective:

```text
L_base = mean(||s_theta(x_t, t) - (-epsilon / sigma_t)||^2)
```

This is a valid Track A starter, but it is not yet strong enough for a higher-confidence ICASSP submission. The current weakness is not only a parameter issue:

- KITTI/PV-RCNN E2.2 Perturbation improves from no-defense, but the absolute row remains low: **11.23 / 6.26**.
- `t_star=0.10` gives only a PV-RCNN Car tradeoff and weakens cross-detector consistency.
- Sparse classes remain fragile: Pedestrian and Cyclist mAP are much lower than Car under PV-RCNN Perturbation.
- The current loss only teaches clean-patch denoising. It does not explicitly preserve local geometry, local density, or sparse-object support.

Therefore, the next improvement should be a loss-function and training-data upgrade rather than another inference-only micro-sweep.

## Research Question

Can a geometry-aware local score-net loss improve detector-transfer robustness under perturbation without increasing point deletion or over-smoothing sparse classes?

## Proposed Loss Variants

### L0 — Baseline DSM

Current training objective. This remains the control checkpoint:

- checkpoint: `checkpoints/kitti/asap_score_net_trackA_v1.pth`
- known best inference setting on KITTI E2.2: `score_anchor_blend=0.75`, `t_star=0.08`
- PointPillars: **61.26 / 34.76**
- PV-RCNN: **11.23 / 6.26**

### L1 — Time-balanced DSM

Problem: the current score target magnitude scales as `1 / sigma_t`, so small-noise timesteps can dominate gradients.

Add a configurable weighting:

```text
L_time = mean(w(t) * ||s_theta(x_t, t) - (-epsilon / sigma_t)||^2)
w(t) in {1, sigma_t^2, clamp(sigma_t^2)}
```

Expected effect: less overfitting to tiny local displacements and more stable denoising at the inference truncation times actually used by ASAP.

First run:

- `loss_profile=time_sigma2`
- same training data size as Track A v1
- same inference setting as `tstar008`

### L2 — Local geometry consistency loss

Problem: score matching alone does not explicitly preserve local shape descriptors used by M2 and by downstream detectors.

Add an auxiliary reconstruction-side loss after the one-step Tweedie estimate:

```text
x_hat = (x_t + sigma_t^2 * s_theta(x_t,t)) / sqrt(alpha_bar_t)
L_geo = L_chamfer(x_hat, x_0)
      + lambda_centroid * ||mean(x_hat) - mean(x_0)||_2^2
      + lambda_cov * ||Cov(x_hat) - Cov(x_0)||_F^2
```

Use a lightweight symmetric Chamfer inside fixed-size patches. The centroid/covariance terms protect sparse local surfaces from collapsing into the local anchor.

First run:

- `loss_profile=geo`
- start with `lambda_chamfer=0.1`, `lambda_centroid=0.05`, `lambda_cov=0.05`
- evaluate only KITTI E2.2 on PointPillars first; continue to PV-RCNN only if PointPillars mAP is not worse than L0 by more than 0.2.

### L3 — Density-ratio preservation loss

Problem: E5.2 showed density ratio is the dominant perturbation cue. A score-net that changes local density too freely can help Car AP while hurting sparse classes.

Approximate local density preservation in normalized patches:

```text
L_density = ||knn_dist_k(x_hat) - knn_dist_k(x_0)||_1
```

Use k in `{4, 8}`. This avoids differentiating through the full M1/M2 scorer while still encouraging local point spacing to stay detector-useful.

First run:

- `loss_profile=geo_density`
- add `lambda_density=0.05`
- evaluate against L2. Promote only if Pedestrian/Cyclist mAP do not regress.

### L4 — Attack-aware paired denoising loss

Problem: training only on clean noised patches does not expose the score-net to actual adversarial perturbation patterns.

Construct paired patches from clean KITTI and attacked E2.2 scans using the same SPU centers, then train a small fraction of batches with an attacked-input target:

```text
L_pair = ||x_hat(attacked_patch, t) - x_clean_patch||_Huber
```

Use this as fine-tuning, not from-scratch training. It is no longer pure generative score matching, but it is still detector-agnostic because it uses clean/attacked geometry pairs rather than detector gradients.

First run:

- `loss_profile=geo_density_pair`
- fine-tune from the best of L1-L3 for 5 epochs
- pair fraction: 25%
- evaluate KITTI E2.2 on both detectors.

## Experiment Gates

### Gate A — Training sanity

Before full purification:

- training loss finite for 1 smoke epoch on CPU or one GPU;
- checkpoint metadata records `loss_profile` and all lambda values;
- 8-frame purification smoke reports nonzero `n_score_edited_points`;
- mean displacement does not exceed L0 by more than 2x.

### Gate B — PointPillars screen

Run KITTI E2.2 PointPillars first:

| Criterion | Promote to PV-RCNN? |
|-----------|----------------------|
| Car AP improves and mAP does not drop | yes |
| mAP improves even if Car is flat | yes |
| Car improves but mAP drops by >0.3 | no, likely sparse-class regression |
| Both Car and mAP drop | no |

### Gate C — PV-RCNN confirmation

Only run PV-RCNN for variants that pass Gate B. A variant is paper-useful only if it improves at least one of:

- PV-RCNN mAP over **6.26** without lowering PointPillars mAP;
- PV-RCNN Car AP over **11.23** while keeping PV-RCNN mAP within `-0.05`;
- Pedestrian/Cyclist mAP improves enough to support a sparse-class robustness claim.

## Execution Order

1. Implement configurable loss profiles in `scripts/train_vpsde_score_net.py`.
2. Add metadata fields to checkpoints:
   - `loss_profile`
   - `lambda_chamfer`
   - `lambda_centroid`
   - `lambda_cov`
   - `lambda_density`
   - `pair_fraction`
3. Train L1 with two T4 GPUs in tmux.
4. Run E2.2 purification with two T4 GPUs in tmux.
5. Evaluate PointPillars with two T4 GPUs.
6. Decide whether to run PV-RCNN.
7. If L1 fails, implement L2. If L1 passes, compare L1 vs L2 before L3.
8. Use L4 only after L1-L3 identify a stable geometry-aware base.

## Commands Template

Training:

```bash
tmux new-session -d -s asap_loss_l1_train_$(date +%Y%m%d_%H%M%S) \
  "cd /root/autodl-tmp/ASAP && \
   CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src \
   /root/miniconda3/envs/asap/bin/python scripts/train_vpsde_score_net.py \
     --clean_dir data/kitti/training/velodyne \
     --out_ckpt checkpoints/kitti/asap_score_net_loss_l1_time_sigma2.pth \
     --max_frames 256 --patches_per_frame 64 --epochs 20 \
     --batch_size 128 --device cuda --data_parallel \
     --loss_profile time_sigma2 \
   > outputs/loss_upgrade/l1_train.log 2>&1"
```

Purification:

```bash
tmux new-session -d -s asap_loss_l1_e22_purify_$(date +%Y%m%d_%H%M%S) \
  "cd /root/autodl-tmp/ASAP && \
   GPUS=0,1 NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 \
   OUT_BASE=outputs/asap_loss_l1_time_sigma2 \
   SCORE_ANCHOR_BLEND=0.75 T_STAR=0.08 \
   SCORE_INJECTION_FILTER_RADIUS=0.0 \
   SCORE_INJECTION_FILTER_MIN_NEIGHBORS=0 \
   SCORE_FILTER_ATTACK_KINDS=injection \
   bash scripts/run_asap_vpsde_score_net_dual_gpu.sh \
     checkpoints/kitti/asap_score_net_loss_l1_time_sigma2.pth \
     E2.2_perturbation \
   > outputs/loss_upgrade/l1_purify.log 2>&1"
```

Evaluation:

```bash
tmux new-session -d -s asap_loss_l1_pp_eval_$(date +%Y%m%d_%H%M%S) \
  "cd /root/autodl-tmp/ASAP && \
   NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 DIST_PORT=19340 EVAL_WORKERS=4 \
   bash scripts/eval_attacked_kitti.sh \
     outputs/asap_loss_l1_time_sigma2/kitti/E2.2_perturbation \
     cfgs/kitti_models/pointpillar.yaml \
     /root/autodl-tmp/ASAP/checkpoints/kitti/pointpillar_7728.pth \
     E_loss_l1_time_sigma2_E2.2_pp 4 && \
   python scripts/parse_eval_summary.py --model pointpillar \
     --out outputs/asap_loss_l1_time_sigma2/kitti_pointpillars_eval_summary.json \
     --entry 'Loss L1 time_sigma2 E2.2:E_loss_l1_time_sigma2_E2.2_pp' \
   > outputs/loss_upgrade/l1_pp_eval.log 2>&1"
```

## Paper Claim Boundary

If a loss variant improves results, the paper can claim a stronger M3 contribution:

> A geometry-regularized local score objective improves detector-transfer purification while preserving sparse-object support.

If no variant improves, the negative result is still useful:

> The current bottleneck is not only the score objective; perturbation robustness likely needs a stronger anomaly scorer or detector-aware-but-weight-frozen validation signal.

Do not claim a new loss contribution until both PointPillars and PV-RCNN have passed Gate C.
