# Post-tstar Reviewer Experiment Decision — 2026-06-04

## Scope

This note closes the KITTI E2.2 `t_star` Track B sweep and converts the latest detector results into the next paper-facing plan. It follows the reviewer-risk logic in `docs/11_icassp_reviewer_readiness_audit.md` and the experiment evidence in `docs/experiments/E5_ablations/E5.6_kitti_pvrcnn_trackB_pfilter.md`.

## Latest Result

The `t_star=0.10` follow-up has finished on both KITTI detectors:

| Setting | Detector | Car Mod AP_R40 | Moderate mAP | Decision signal |
|---------|----------|---------------:|-------------:|-----------------|
| `tstar008` | PV-RCNN | **11.23** | **6.26** | current paper-facing row |
| `tstar010` | PV-RCNN | 11.36 | 6.16 | Car improves, mAP drops |
| `tstar008` | PointPillars | **61.26** | **34.76** | current paper-facing row |
| `tstar010` | PointPillars | 61.06 | 34.76 | Car drops, mAP is only tied after rounding |

Interpretation: `tstar010` is a class/detector tradeoff, not a robust improvement. It slightly improves PV-RCNN Car AP but weakens cross-class mAP and PointPillars Car AP. Because the paper claims detector-agnostic and cross-class robustness, `tstar008` remains the KITTI Perturbation candidate.

## Experiment Completeness

The current experiment matrix is complete enough for a conservative ICASSP submission if the manuscript avoids broad SOTA-dominance claims.

Completed acceptance-critical evidence:

- KITTI two-detector matrix: PointPillars and PV-RCNN are covered for Injection, Perturbation, and Dropping with no-defense, SOR/ROR, and ASAP-family policies.
- nuScenes two-detector matrix: VoxelNeXt and TransFusion-Lidar are covered for accepted Injection and Perturbation rows; Dropping is excluded because the keyframe-only attack failed the strength gate under ten-sweep evaluation.
- Recent-method context: the LiDAR-SPD-style fixed spherical proxy is completed and unfavorable to nonselective diffusion, while PWAVEP/CCS 2025 are documented as not directly same-protocol full-scene KITTI baselines.
- Core ablations: M1 fixed-radius diagnostic, M2 feature diagnostic, M2 all-unit collapse, tau sensitivity, pfilter/support-only comparison, qualitative selectivity, and cost/selective-ratio summaries are all available.

Remaining risk is not "missing one more `t_star` run." The main risk is reviewer perception:

1. `ASAP-pfilter` is close to ROR unless the cascade boundary is kept explicit.
2. Recent SOTA methods are not officially reproduced under full-scene detection.
3. nuScenes attacks are keyframe-only under ten-sweep evaluation.
4. PV-RCNN Perturbation improves clearly over no-defense/ROR but remains low in absolute AP.

## Decision

Freeze the current main experimental rows:

- KITTI PointPillars Perturbation: `tstar008`, **61.26 / 34.76**.
- KITTI PV-RCNN Perturbation: `tstar008`, **11.23 / 6.26**.
- Do not promote `tstar010`.
- Do not continue `t_star` micro-sweeps unless a new objective is introduced, such as class-aware sparse-object mAP or a learned scorer revision.

The next work should shift from parameter tuning to reviewer-readiness.

## Next Work Plan

### P0 — Reviewer-facing consistency audit

Goal: make sure the final paper says exactly what the experiments prove.

Actions:

1. Check that Abstract, Introduction, Method, Experiments, and Conclusion all describe `ASAP-pfilter` as a score-net + support-filter cascade.
2. Check that KITTI `tstar008` is described as a calibrated inference-time setting, not a new method module.
3. Check that Dropping is framed as conservative damage avoidance, not reconstruction.
4. Check that no sentence claims broad SOTA dominance over all recent methods.

Deliverable: update `docs/13_markdown_reviewer_pass.md` or create a short post-freeze addendum.

### P1 — Optional single targeted experiment only if the audit demands it

Do not launch a broad new matrix. If one more experiment is necessary, choose exactly one of:

| Candidate | Why it helps | Cost/risk | Current recommendation |
|-----------|--------------|-----------|------------------------|
| Class-aware sparse-object diagnostic on KITTI E2.2 | Addresses Pedestrian/Cyclist weakness and PV-RCNN low mAP | Requires scorer/policy change; may not finish cleanly | Defer unless reviewer audit flags sparse classes as fatal |
| Qualitative pfilter-vs-ROR panels on nuScenes Perturbation | Helps novelty explanation for the pfilter cascade | Mostly analysis/visualization, low GPU cost | Best optional add-on |
| Official external baseline reproduction | Strongest SOTA response if runnable full-scene code exists | High uncertainty; likely code/protocol mismatch | Do not start without a found runnable baseline |
| More `t_star` values | Could find a tiny AP point | Low scientific value; risks overfitting | Stop |

Preferred optional add-on: qualitative pfilter-vs-ROR visual/selectivity analysis, because it directly reduces the "is this just ROR?" reviewer risk without changing the main claim.

### P2 — Manuscript assembly later

The user preference is to keep LaTeX for the end. Therefore the next manuscript step is Markdown claim polishing and reviewer-pass updates, not LaTeX assembly.

## Go / No-Go

Go for paper-facing reviewer-readiness work. No-go for additional `t_star` sweeps.

Current status: **main experiments frozen, with optional low-cost qualitative support analysis if needed**.
