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
| P0 | L1b `anchor085` purification | done | Purified KITTI E2.2 with L1 checkpoint, `score_anchor_blend=0.85`, `t_star=0.08`, GPU 1 only | Produced 3769 frames and 3769 meta rows |
| P0 | L1b `anchor085` PointPillars gate | failed | Evaluated L1b purified split with PointPillars on GPU 1 | Failed: 60.4571 / 34.1900 |
| P0 | L1c `step07` purification | done | tmux `asap_l1c_step07_e22_purify_gpu1_20260609_000556`; L1 checkpoint, `score_step_size=0.7`, anchor blend 0.75, GPU 1 only | Produced 3769 frames and 3769 meta rows |
| P0 | L1c `step07` PointPillars gate | failed | Evaluated L1c purified split with PointPillars on GPU 1 | Failed: 61.1926 / 34.2711 |
| P1 | L1c PV-RCNN confirmation | skipped | PointPillars gate failed | Do not spend PV-RCNN time on this branch |
| P0 | L1d edit-vote smoke | failed | Ran `policy_min_votes=2` 8-frame metadata diagnostic on GPU 1 | Failed: edited points collapsed from 23.05% to 2.51% on matched frames |
| P1 | L1d edit-vote full gate | skipped | Smoke coverage was too conservative | Do not run full purification/eval |
| P0 | R1 fixed-r040 PV-RCNN confirmation | failed | Evaluated existing E5.1 fixed `r=0.40` full output with PV-RCNN on GPU 1 | Failed: 10.9612 / 6.1344 |
| P1 | R2 L1 fixed-r040 purification | skipped | R1 did not transfer to PV-RCNN | Do not spend a full purification on this radius branch |
| P0 | L4 paired fine-tune implementation | done | Implemented `geo_density_pair`, `--paired_attacked_dir`, `--pair_fraction`, `--lambda_pair`, and `--init_ckpt` | GPU1 smoke passed and checkpoint metadata is correct |
| P0 | L4 paired fine-tune training | active | Fine-tune from L1 on clean + E2.2 paired patches using GPU 1 in tmux | Train 5 epochs, finite losses, checkpoint metadata records paired settings |
| P0 | L4 E2.2 purification | pending | Purify KITTI E2.2 using L4 checkpoint on GPU 1 | Produce 3769 frames and 3769 meta rows |
| P0 | L4 PointPillars gate | pending | Evaluate L4 purified split on GPU 1 | Pass if mAP >= 34.7457 or sparse classes improve with mAP within -0.05 |
| P3 | L3b low-weight density | downgraded | Train `density` with `lambda_density=0.01` only if inference-side checks also fail | Current L3 failed by large mAP drop, so do not prioritize more density training |
| P3 | Paired fine-tune | promoted | Promoted to L4 after L1b/L1c/L1d/R1 all failed | Detector-agnostic attacked-clean geometry pairing is the next distinct mechanism |

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

### L1b `anchor085` GPU1-only purification

- tmux session: `asap_l1b_anchor085_e22_purify_gpu1_20260608_230520`.
- Output root: `outputs/asap_l1b_anchor085/kitti/E2.2_perturbation/`.
- Settings: L1 checkpoint, `score_anchor_blend=0.85`, `t_star=0.08`, `GPUS=1`, `CUDA_VISIBLE_DEVICES=1`.
- Result: **3769 / 3769** purified frames and **3769** metadata rows.
- Metadata ratios: **27.78%** flagged SPUs, **23.71%** score-net SPUs, **21.93%** edited points, no support-filter drops.
- Status: purification succeeded; PointPillars gate is active.

### L1b `anchor085` PointPillars gate

- Result: PointPillars **60.4571 / 34.1900**.
- Comparison to L1: Car **-0.8933**, mAP **-0.5557**, Pedestrian **-0.9679**, Cyclist **+0.1940**.
- Decision: failed gate; PV-RCNN not run.
- Interpretation: higher anchor blend is too conservative. It slightly helps Cyclist but harms Car and Pedestrian enough to make the variant unusable.
- Plan correction: try `L1c_step07`, keeping anchor blend at 0.75 but reducing `score_step_size` to 0.7 to shrink score displacement more evenly.

### L1c `step07` PointPillars gate

- Purification: **3769 / 3769** frames and **3769** metadata rows.
- Metadata ratios: **27.78%** flagged SPUs, **23.71%** score-net SPUs, **21.93%** edited points, same coverage as L1/L1b/L3.
- Result: PointPillars **61.1926 / 34.2711**.
- Comparison to L1: Car **-0.1578**, mAP **-0.4746**, Pedestrian **-1.1019**, Cyclist **-0.6541**.
- Decision: failed gate; PV-RCNN not run.
- Interpretation: reducing step size does not protect sparse-class geometry. Because coverage is unchanged, the failure is not from selecting fewer or more SPUs; it is from the edited point subset still receiving harmful coordinate updates.
- Plan correction: stop same-family anchor/step sweeps and move to L1d `edit_vote2`, a targeted test of whether requiring overlapping flagged SPUs can reduce harmful edits.

### L1d `edit_vote2` smoke

- Smoke setup: 8 KITTI E2.2 frames, L1 checkpoint, `score_anchor_blend=0.75`, `t_star=0.08`, `policy_min_votes=2`, GPU 1 only.
- Matched-frame L1/L1c coverage: **29.30%** flagged SPUs, **25.88%** score-net SPUs, **23.29%** edited points, **23.05%** score-edited points.
- L1d smoke coverage: **29.30%** flagged SPUs, **10.10%** score-net SPUs, **2.58%** edited points, **2.51%** score-edited points.
- Decision: failed smoke; do not run full purification.
- Interpretation: requiring two flagged SPU votes is not a moderate support-aware correction. It removes almost all edits and would likely repeat the stricter-`tau` under-editing failure.
- Plan correction: move away from vote thresholding. The best existing PointPillars mAP evidence is E5.1 fixed `r=0.40` (**61.0645 / 34.9196**), so run PV-RCNN confirmation on that already-complete output before spending time on new full purification.

### R1 fixed `r=0.40` PV-RCNN confirmation

- PointPillars prior from E5.1: **61.0645 / 34.9196**, Pedestrian **12.0363**, Cyclist **31.6580**.
- PV-RCNN result: **10.9612 / 6.1344**, Pedestrian **0.6860**, Cyclist **6.7560**.
- Comparison to L1 PV-RCNN: Car **-0.2722**, mAP **-0.3225**, Pedestrian **+0.0314**, Cyclist **-0.7268**.
- Decision: failed transfer; do not run R2 L1 fixed-r040 full purification.
- Interpretation: fixed `r=0.40` is a PointPillars mAP tradeoff and remains useful as an E5.1 ablation, but it does not solve the cross-detector ICASSP risk.
- Plan correction: move to L4 attacked-clean paired fine-tuning, which changes training exposure rather than editing magnitude/radius.

### L4 `geo_density_pair` implementation smoke

- Code: `scripts/train_vpsde_score_net.py`.
- New options: `loss_profile=geo_density_pair`, `--paired_attacked_dir`, `--paired_max_frames`, `--paired_patches_per_frame`, `--pair_fraction`, `--lambda_pair`, `--init_ckpt`.
- Smoke command: GPU1, 1 clean frame, 1 paired E2.2 frame, 4 patches/frame, 1 epoch, initialized from L1.
- Smoke checkpoint: `checkpoints/kitti/asap_score_net_loss_l4_pair_smoke.pth`.
- Result: finite loss **0.335084**, DSM **0.332419**, weighted density **0.000987**, raw density **0.098709**, pair **0.006712**.
- Metadata: `loss_profile=geo_density_pair`, `pair_fraction=0.25`, `lambda_pair=0.25`, `lambda_density=0.01`, `init_ckpt=...loss_l1_time_sigma2.pth`.
- Decision: implementation passed; start full GPU1-only L4 fine-tune.

## Conditional Breakthrough Queue

### L1d `edit_vote2`

- Trigger: run only if L1c fails the PointPillars gate.
- Diagnosis targeted: L2/L3/L1b suggest the main failure is not insufficient denoising capacity; it is likely editing the wrong subset or over-stabilizing useful sparse evidence.
- Code status: `PurifierConfig.policy_min_votes` already exists, but the CLI does not expose it for VP-SDE runs.
- Code change: `--policy_min_votes` and `POLICY_MIN_VOTES` are now exposed for VP-SDE scripts, with default behavior unchanged.
- Experiment shape: first run an 8-frame smoke/metadata diagnostic with `policy_min_votes=2`; only launch full KITTI E2.2 purification if edited-point ratio drops moderately without collapsing score-SPU coverage.
- Risk: stricter `tau` already failed badly, so this must be treated as a targeted coverage diagnostic, not as a broad "edit less" sweep.

### R1 `fixed_r040_pvrcnn`

- Trigger: active after L1d smoke failed.
- Rationale: E5.1 fixed `r=0.40` already gives the best available KITTI PointPillars mAP among nearby mechanisms: **34.9196**, with Pedestrian **12.0363** and Cyclist **31.6580**.
- Cost advantage: no new purification needed for R1; reuse `outputs/ablations/E5.1_fixed_radius/fixed_r040/kitti/E2.2_perturbation`.
- Risk: it may be a PointPillars-only mAP tradeoff because Car AP is lower than L1 and the fixed radius edits more points.
- Decision rule: if PV-RCNN improves over **6.4569** mAP, promote fixed-radius combination for a stronger follow-up; otherwise treat fixed `r=0.40` as an ablation-only diagnostic.

### L4 `geo_density_pair`

- Trigger: active after R1 failed cross-detector transfer.
- Rationale: L1b/L1c changed displacement magnitude, L1d changed edit coverage, and R1 changed radius; none produced a cross-detector improvement. L4 targets the remaining likely gap: the score-net has never seen actual attacked geometry during training.
- Training plan: initialize from L1, use `loss_profile=geo_density_pair`, `pair_fraction=0.25`, `lambda_pair=0.25`, `lambda_density=0.01`, 5 epochs, GPU 1 only.
- Gate: run PointPillars first. PV-RCNN only if PointPillars mAP is not worse than L1 by more than **0.05** or sparse-class AP clearly improves.

## Next Candidate Design Rules

- If PointPillars mAP drops by more than **0.05**, do not run PV-RCNN for that variant.
- If a loss variant fails, the next variant must directly target the diagnosed failure mode.
- Avoid stacking auxiliary losses after L3 unless PointPillars suggests a real sparse-class gain.
- Prefer low-weight or inference-side checks over increasingly strong reconstruction objectives.
- After L1c, do not continue same-family anchor/step sweeps unless detector metrics identify a specific class tradeoff worth isolating.
