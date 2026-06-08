# Dynamic Experiment Worklist

## Operating Rules

- Hardware rule: use **GPU 1 only** for all future training, purification, and evaluation.
- Long-running jobs must run in `tmux`.
- After every failed experiment, record the failure reason, update the next plan, and add a new candidate only if it addresses the observed failure.
- After every successful gate, immediately document the setting, results, interpretation, commit the record, and push.
- Paper-facing claims stay frozen until a variant passes the required detector gates.

## Current Baseline

- Main E7 candidate: L1 `time_sigma2`.
- KITTI E2.2 PointPillars: **61.3504 / 34.7457**.
- KITTI E2.2 PV-RCNN: **11.2334 / 6.4569**.
- Current paper table should not be updated until a stronger variant passes both detector gates.

## Active Worklist

| Priority | ID | Status | Action | Gate |
|----------|----|--------|--------|------|
| P0 | L3 `density` GPU1-only purification | done | Restarted KITTI E2.2 purification using only GPU 1; old dual-GPU partial output discarded | Produced 3769 frames and 3769 meta rows |
| P0 | L3 PointPillars gate | failed | Evaluated `outputs/asap_loss_l3_density/kitti/E2.2_perturbation` with PointPillars on GPU 1 | Failed: 61.1388 / 34.3561 |
| P0 | L1b `anchor085` purification | active | Purify KITTI E2.2 with L1 checkpoint, `score_anchor_blend=0.85`, `t_star=0.08`, GPU 1 only | Produce 3769 frames and 3769 meta rows |
| P0 | L1b `anchor085` PointPillars gate | pending | Evaluate L1b purified split with PointPillars on GPU 1 | Pass if mAP >= 34.7457 or sparse-class AP recovers with mAP within -0.05 |
| P1 | L1b PV-RCNN confirmation | conditional | Run only if PointPillars gate passes | Improve PV-RCNN mAP over 6.4569, or improve Pedestrian while keeping mAP near L1 |
| P2 | L1c `step07` inference check | candidate | If L1b fails, purify with L1 checkpoint, `score_step_size=0.7`, anchor blend 0.75 | Test whether smaller score displacement protects sparse classes |
| P3 | L3b low-weight density | downgraded | Train `density` with `lambda_density=0.01` only if inference-side checks also fail | Current L3 failed by large mAP drop, so do not prioritize more density training |
| P3 | Paired fine-tune | deferred | Fine-tune from L1/L3 only after a stable loss base exists | Avoid detector-aware claims until clean evidence exists |

## Failure Log

### L2 `geo`

- Result: PointPillars **61.2623 / 34.5159**.
- Comparison to L1: Car **-0.0881**, mAP **-0.2298**.
- Decision: failed PointPillars gate; PV-RCNN not run.
- Interpretation: Chamfer-heavy local reconstruction likely over-constrained the score field and hurt detector-useful sparse-class geometry.
- Plan correction: move to L3 density-only spacing preservation without Chamfer/centroid/covariance.

### L3 partial purification from 2026-06-05

- Output: old run created partial shard outputs only, about 992 `.bin` files and 980 shard meta rows.
- Issue: run used `GPUS=0,1`, which violates the current GPU1-only rule, and did not finish with a merged `meta.jsonl`.
- Decision: discard this partial output and restart L3 purification with `GPUS=1`, `NUM_GPUS=1`, `CUDA_VISIBLE_DEVICES=1`.

## Success Log

### L3 `density` training

- Checkpoint: `checkpoints/kitti/asap_score_net_loss_l3_density.pth`.
- Training: 256 KITTI frames, 64 patches/frame, **16384** patches, 20 epochs, 2560 steps.
- Loss profile: `density`, `lambda_density=0.03`, `density_k=8`.
- Final loss: **0.371031**.
- Final breakdown: DSM **0.368540**, weighted density **0.002491**, raw density **0.083023**.
- Status: training succeeded; downstream purification/evaluation still pending.

### L3 GPU1-only purification

- tmux session: `asap_loss_l3_e22_purify_gpu1_20260608_215700`.
- Output root: `outputs/asap_loss_l3_density/kitti/E2.2_perturbation/`.
- GPU rule: `GPUS=1`, `NUM_GPUS=1`, `CUDA_VISIBLE_DEVICES=1`.
- Result: **3769 / 3769** purified frames and **3769** metadata rows.
- Metadata ratios: **27.78%** flagged SPUs, **23.71%** score-net SPUs, **21.93%** edited points, no support-filter drops.
- Status: purification succeeded; PointPillars gate is active.

## Latest Failure Analysis

### L3 `density` PointPillars gate

- Result: PointPillars **61.1388 / 34.3561**.
- Comparison to L1: Car **-0.2116**, mAP **-0.3896**, Pedestrian **-0.5449**, Cyclist **-0.4123**.
- Decision: failed gate; PV-RCNN not run.
- Interpretation: after L2 and L3 both reduced mAP, auxiliary training losses are likely preserving proxy geometry while degrading detector-useful perturbation recovery.
- Plan correction: stop stacking loss functions for now. Move to inference-side conservative updates using the L1 checkpoint:
  - next active candidate: `L1b_anchor085`, increasing `score_anchor_blend` from 0.75 to 0.85;
  - backup candidate: `L1c_step07`, reducing `score_step_size` from 1.0 to 0.7.

## Next Candidate Design Rules

- If PointPillars mAP drops by more than **0.05**, do not run PV-RCNN for that variant.
- If a loss variant fails, the next variant must directly target the diagnosed failure mode.
- Avoid stacking auxiliary losses after L3 unless PointPillars suggests a real sparse-class gain.
- Prefer low-weight or inference-side checks over increasingly strong reconstruction objectives.
