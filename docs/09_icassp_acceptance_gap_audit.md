# ICASSP Acceptance Gap Audit — 2026-05-31

## Goal

Maximize the chance that the ASAP ICASSP submission survives reviewer scrutiny. The immediate priority is not more raw experiments, but a manuscript whose claims, method description, and tables are internally consistent and defensible.

## Current Submission-Critical Evidence

| Claim | Evidence status | Risk |
|-------|-----------------|------|
| Inference-only and detector-agnostic front-end | Supported across PointPillars, PV-RCNN, VoxelNeXt, and TransFusion-Lidar. | Low, if detector checkpoints and no-retraining protocol stay explicit. |
| Injection robustness | ASAP-v1 is near ROR but not consistently above it. | Medium. Phrase as near-ROR recovery, not dominance. |
| Perturbation robustness | ASAP-pfilter beats ROR on both nuScenes detectors: VoxelNeXt 52.27/61.23 vs 51.36/60.40; TransFusion 54.40/61.98 vs 53.51/61.23. | Medium. Must label pfilter as score-net + support-filter cascade. |
| Dropping robustness | KITTI uses conservative skip-M3; nuScenes Dropping excluded after failing attack-strength gate. | Medium. Do not claim point restoration. |
| Modern detector transfer | Supported on VoxelNeXt and TransFusion-Lidar under accepted nuScenes attacks. | Low-medium because nuScenes attacks are keyframe-only with clean sweeps. Caveat must remain visible. |
| Method novelty over ROR | Supported by selective SPU scoring + score-net denoising, but pfilter uses a ROR-like support filter. | Medium-high. Need explain pfilter as a controlled cascade, not a hidden baseline reuse. |
| Novelty versus recent SOTA | Strengthened by fixed-SPU all-unit diffusion and adaptive-SPU all-unit ablations, but still lacks official full reproductions of recent defenses. | Medium-high. Use these as proxy ablations; avoid claiming broad SOTA dominance. |

## Reviewer-Risk Notes

1. A reviewer may object that ASAP-pfilter is too close to ROR. The response is that ROR is applied globally as a classical baseline, while ASAP-pfilter applies score-net denoising first and then a clearly disclosed support filter. The paper must avoid saying that learned diffusion alone beats ROR on nuScenes Perturbation.
2. A reviewer may object to attack-aware policies. The paper should frame this as an attack-family prior that prevents known failure modes: editing helps coordinate perturbation, support filtering removes sparse artifacts, and dropping should not be edited when evidence has already been removed.
3. A reviewer may object to nuScenes keyframe-only attacks because OpenPCDet uses 10 sweeps. The paper must state that historical sweeps remain clean and that Dropping is excluded for this reason.
4. A reviewer may object to runtime. The current prototype is slower than ROR, but detector-agnostic and no-retraining. Keep the claim about robustness and transfer, not real-time deployment.

## Immediate Next Actions

1. Treat the P0/P1/P2 novelty gap as resolved enough for a first submission draft:
   - Fixed-SPU all-unit diffusion: **57.16 / 30.35**, much heavier and below ASAP Track A.
   - Adaptive-SPU all-unit diffusion: **38.62 / 17.88**, showing over-purification collapse without M2.
   - Support-only/ROR vs ASAP-pfilter: pfilter exceeds the same support rule by about +0.9 mAP on both nuScenes Perturbation detectors.
2. Treat the tau sensitivity branch as closed. Stricter thresholds (`tau=0.036`, `tau=0.048`) reduce KITTI Perturbation mAP versus the current `tau=0.024`, so the final method should not tune thresholds further for this submission.
3. Assemble and audit the first IEEE-style manuscript. The next acceptance-critical task is claim control and reviewer-risk reduction, not another broad experiment matrix.
4. If the review pass requests one more experiment, prefer a qualitative visualization of selected SPUs / pfilter effects or an external-SOTA feasibility note over new large-scale detector runs.

## Cleanup Completed on 2026-05-31

- Removed submission-facing TODO/placeholder/HTML-comment artifacts from `docs/paper/00_abstract.md` through `05_conclusion.md`.
- Replaced the `cite-PointGuard` placeholder with a verified PointGuard citation and added detector citations for PointPillars, PV-RCNN, VoxelNeXt, and TransFusion.
- Rewrote `docs/paper/references.bib` to remove `UNVERIFIED` notes and keep all cited BibTeX keys resolvable.
- Added LaTeX-ready Figure 1 and Tables 1-2 in `docs/paper/fig_asap_pipeline.tex` and `docs/paper/tables_icassp.tex`.
- Added `scripts/check_paper_integrity.py`; current check passes with no missing BibTeX keys or submission-facing placeholders.
