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
| P0 | L3 `density` GPU1-only purification | active | Restart KITTI E2.2 purification using only GPU 1; discard old dual-GPU partial output | Produce 3769 frames and 3769 meta rows |
| P0 | L3 PointPillars gate | pending | Evaluate `outputs/asap_loss_l3_density/kitti/E2.2_perturbation` with PointPillars on GPU 1 | Pass if mAP >= 34.6957 and at least one sparse class improves, or if mAP improves over L1 |
| P1 | L3 PV-RCNN confirmation | conditional | Run only if PointPillars gate passes | Improve PV-RCNN mAP over 6.4569, or improve Pedestrian while keeping mAP near L1 |
| P2 | L3b low-weight density | candidate | If L3 fails by small mAP drop, train `density` with `lambda_density=0.01` | Test whether current density weight is still too restrictive |
| P2 | L1+inference micro-check | candidate | If L3 fails, run a small inference-only check around L1 with unchanged training | Avoid adding more loss complexity if training objectives saturate |
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

## Next Candidate Design Rules

- If PointPillars mAP drops by more than **0.05**, do not run PV-RCNN for that variant.
- If a loss variant fails, the next variant must directly target the diagnosed failure mode.
- Avoid stacking auxiliary losses after L3 unless PointPillars suggests a real sparse-class gain.
- Prefer low-weight or inference-side checks over increasingly strong reconstruction objectives.
