# ASAP Innovation and SOTA Upgrade Plan — 2026-05-31

## Why the Current Paper Is Still at Risk

The current ASAP evidence is defensible but not yet strong enough for a confident ICASSP submission:

1. **Innovation risk.** ASAP-pfilter can be interpreted as score-net denoising followed by a ROR-like support filter. If this is the only row that beats ROR on nuScenes Perturbation, reviewers may question whether the method contribution is larger than a tuned classical filter.
2. **SOTA baseline risk.** The current main table mainly compares against SOR/ROR. That is not enough when recent LiDAR robustness papers include LiDAR-SPD, LiDARPure, ScAR, PCLD, PointDP, and Ada3Diff.
3. **Ablation risk.** The manuscript needs direct evidence that M1 adaptive radius, M2 anomaly-guided selection, and M3 score-net purification each matter.
4. **Threat-model risk.** nuScenes attacks are keyframe-only under a 10-sweep OpenPCDet evaluation pipeline. This is valid if disclosed, but it weakens claims about full temporal LiDAR robustness.

## Revised Positioning

The paper should not try to claim broad SOTA dominance. A stronger and more honest target is:

> ASAP is a detector-agnostic, inference-only LiDAR purification framework whose novelty is **adaptive local purification with anomaly-guided compute allocation**. It is competitive with strong classical support filtering, improves over fixed/global spherical diffusion-style baselines, and transfers across modern nuScenes detectors under accepted keyframe-only attacks.

This positioning requires one new baseline family and two tighter ablations.

## Method Upgrade: ASAP-Support

Current pfilter should be reframed and upgraded from an afterthought into a named, principled branch:

- **Current:** score-net output followed by radius filter.
- **Upgrade name:** Support-Aware Selective Purification, or **ASAP-Support**.
- **Core idea:** use support consistency as a *local post-denoising confidence criterion* inside the selected SPUs, not as a global scan denoiser.
- **Claim boundary:** ASAP-Support is a cascade. It should be compared directly against global ROR and fixed-radius uniform diffusion to show why local selection matters.

Minimum method changes needed:

1. Report support filtering only after M2-selected score-net purification.
2. Add metadata showing filtered points come from a small selected region, not global scan deletion.
3. Add ablation rows:
   - ROR only.
   - score-net only.
   - support filter only, if cheap to produce.
   - score-net + support filter.

## Experiment Priority

### P0 — SPD-style uniform spherical diffusion baseline

Purpose: answer whether ASAP is better than a LiDAR-SPD-style fixed spherical diffusion design.

Implementation:

- Fixed radius: `r1 = 0.40 m`, `r2 = 0.67 r1`.
- Fixed stride: `eta = 0.40 m`.
- Disable anomaly gating with `tau = -1.0`, so every SPU is purified.
- Use the same Track A score-net checkpoint and anchor blend as ASAP.
- Run first on KITTI PointPillars E2.2 Perturbation, where ASAP has a strong result.

Status:

- Code support added on 2026-05-31:
  - `python -m asap.pipeline` now accepts explicit M1 SPU hyperparameters.
  - `scripts/run_spd_style_uniform_score_net_dual_gpu.sh`
  - `scripts/eval_spd_style_uniform_kitti.sh`
- 20-frame smoke result:
  - 100% SPUs flagged by design.
  - 97.45% SPUs entered score-net.
  - 77.81% of points were edited.
  - Single-T4 runtime: 4.74 s/frame.
- Full KITTI E2.2 Perturbation completed and evaluated on 2026-06-01:
  - 3769/3769 KITTI val frames purified.
  - PointPillars result: **57.16 / 30.35** Car/Moderate AP_R40 / cross-class mAP.
  - Full-run selectivity/cost: 100.00% SPUs flagged, 97.14% SPUs entered score-net, 77.86% input points edited.
  - Effective two-T4 preprocessing throughput is about 2.39 s/frame; per-shard logs report about 4.78 s/frame.
  - Output summary: `outputs/spd_style_uniform_score_net/kitti_pointpillars_eval_summary.json`.

Decision rule:

- If accuracy is worse than ASAP while runtime/edit ratio is much higher, this becomes a strong ablation proving M1/M2 selectivity.
- If accuracy is better, the method must be revised around support-aware fixed-unit diffusion because current ASAP is not sufficiently strong.

Decision:

- P0 supports the ASAP selectivity claim. The fixed-SPU all-unit diffusion proxy is slightly above no-defense (**57.16 / 30.35** vs **56.54 / 27.32**) but clearly below ASAP Track A (**61.04 / 34.69**) while editing nearly four-fifths of all points. This is now a key innovation ablation, not a replacement method.

### P1 — Support-filter-only baseline

Purpose: answer whether ASAP-pfilter is merely ROR.

Rows to add for nuScenes Perturbation:

- ROR global: already available and uses the same `radius=0.4, min_neighbors=5` rule as the pfilter branch, so it serves as the support-filter-only row on the attacked input.
- score-net only: already available as ASAP-v1.
- score-net + support filter: already available as ASAP-pfilter.

Decision rule:

- If support-only is equal to pfilter, ASAP-pfilter has weak novelty and should be demoted.
- If score-net + support beats support-only, the cascade has a real interaction effect.

Current outcome:

- VoxelNeXt Perturbation: support-only/ROR **51.36 / 60.40**, score-net only **41.32 / 52.74**, score-net + support **52.27 / 61.23**.
- TransFusion-Lidar Perturbation: support-only/ROR **53.51 / 61.23**, score-net + support **54.40 / 61.98**.
- This shows the support branch is the dominant contributor, but score-net preconditioning still adds a small, detector-transferable gain over support-only. The manuscript must present ASAP-pfilter as a cascade rather than pure diffusion superiority.

KITTI/PV-RCNN correction on 2026-06-03:

- Directly transferring the same support rule to KITTI/PV-RCNN E2.2 is negative: Track B pfilter r04m5 reaches **10.53 / 4.99**, below Track A **10.96 / 5.98**.
- Metadata shows the branch edits **21.93%** of points and then drops another **5.32%**, so the likely issue is over-removal of sparse detector-useful evidence.
- The non-deletion `score_anchor_blend=0.50` variant is also negative: **8.31 / 5.03**, below Track A **10.96 / 5.98**. The local anchor appears stabilizing.
- The modest `t_star=0.08` variant is positive while keeping `score_anchor_blend=0.75`: PV-RCNN E2.2 improves from **10.96 / 5.98** to **11.23 / 6.26**, and PointPillars E2.2 improves from **61.04 / 34.69** to **61.26 / 34.76**.
- The `t_star=0.10` follow-up is not promoted: PV-RCNN Car AP increases to **11.36**, but PV-RCNN mAP drops to **6.16** and PointPillars Car AP drops to **61.06**. This is a class/detector tradeoff, not a robust improvement.
- Decision: keep pfilter as a disclosed nuScenes Perturbation cascade; do not claim it is a universal fix. Use `t_star=0.08` as the current KITTI Perturbation setting, with the caveat that the absolute PV-RCNN recovery remains low. Stop the `t_star` micro-sweep unless a new objective is introduced.

### P2 — M1/M2 ablation table

Purpose: show adaptive radius and anomaly selection are not cosmetic.

Rows:

- Fixed radius + all SPUs: P0, **57.16 / 30.35**.
- Adaptive radius + all SPUs: set `tau=-1.0` with normal M1, **38.62 / 17.88**.
- Adaptive radius + M2 selection: current ASAP, **61.26 / 34.76** with the E2.2 `t_star=0.08` setting.

Outcome:

- P2 strongly supports M2 as a core methodological contribution. Adaptive SPUs without anomaly selection edit **70.98%** of points, send **95.92%** of SPUs through the score-net path, and collapse below both no-defense and the fixed-SPU proxy. The next method revision should foreground M2 as a structure-preservation gate, not only as a compute allocator.

### P3 — External SOTA feasibility audit

Candidate external methods:

| Method | Fit to our task | Feasibility |
|--------|-----------------|-------------|
| LiDAR-SPD | Direct LiDAR detection defense; closest conceptual competitor. | PDF available, no local code found. Use SPD-style proxy unless code is found. |
| LiDARPure | LiDAR scene diffusion purification, common corruptions. | PDF available; attack-defense mismatch with adversarial rows. |
| ScAR | LiDAR detection robustness, but training-time/adversarial training style. | Not inference-only; compare in related work, not direct baseline unless code/checkpoints are feasible. |
| PointDP / PCLD / Ada3Diff | Strong point-cloud purification work. | Mostly classification / object-level point clouds, not full LiDAR detection. Use cautiously as related work or proxy only. |

### P4 — M2 threshold sensitivity

Purpose: test whether stricter anomaly selection can reduce over-purification and improve sparse-class mAP on KITTI PointPillars Perturbation.

Rows:

- Current Track A threshold `tau=0.024`: **61.04 / 34.69**.
- Stricter threshold `tau=0.036`: **59.88 / 30.21**.
- Stricter threshold `tau=0.048`: **58.02 / 28.79**.

Outcome:

- The stricter thresholds reduce both Car AP and cross-class mAP, with the largest damage in sparse classes. The current ROC-calibrated `tau=0.024` remains the final policy.
- This result refines the M2 story: the gate is essential, but manually editing fewer SPUs is not a substitute for a well-calibrated anomaly score. The manuscript should use P0/P2 as the primary novelty ablation and mention P4 only as a supporting sensitivity check.

## Near-Term Execution Plan

1. Keep P0/P2 in Table 2 as the key innovation ablation: nonselective diffusion is either weak or harmful.
2. Treat P4 as closed: keep `tau=0.024`; do not tune thresholds further unless a new scorer or class-aware objective is introduced.
3. Treat the KITTI `t_star` Track B sweep as closed: keep `t_star=0.08` for E2.2 and do not replace it with `0.10`.
4. Revise method wording to foreground M2 as an anomaly-selective structure-preservation gate.
5. Move to manuscript assembly and reviewer-style audit. New GPU experiments should be launched only if the audit identifies a specific acceptance-critical gap.
