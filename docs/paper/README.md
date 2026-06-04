# ASAP ICASSP paper draft

This folder contains the working draft of the ASAP paper, targeting **ICASSP 2026**.
Each section is kept in its own Markdown file so that drafting, review, and Git history stay focused per section. The final LaTeX manuscript will be assembled later from these Markdown drafts.

## Working title

**ASAP: Adaptive Spherical Anomaly-Guided Purification Against LiDAR Point Cloud Attacks**

## Current paper track

**Track A selected on 2026-05-29:** the paper pursues the full ASAP diffusion claim with a trained point-cloud score network. Existing local-mean, drop, and hybrid results are implementation milestones and ablations, not the final ASAP row.

Implementation status on 2026-06-03: a local VP-SDE score-net checkpoint (`checkpoints/kitti/asap_score_net_trackA_v1.pth`) has been trained and evaluated. Adding a local-manifold anchor with `score_anchor_blend=0.75` gives strong learned-M3 numbers on PointPillars E2.2 Perturbation, and the current `t_star=0.08` follow-up reaches **61.26** Car/Mod R40 and **34.76** mAP. Adding an attack-conditional Injection radius filter (`0.4 m`, min 5 neighbors) fixes PointPillars E2.1 Injection to **65.71** Car/Mod R40 and **52.18** mAP. PV-RCNN validation confirms the Injection branch transfers (**72.59** vs **63.16** no-defense Car/Mod R40), and `t_star=0.08` improves PV-RCNN E2.2 to **11.23** Car/Mod R40 and **6.26** mAP, while learned M3 regresses Dropping. The current revised KITTI policy therefore uses b075-ifilter for E2.1, b075 with `t_star=0.08` for E2.2, and skips M3 for E2.3 Dropping, restoring PV-RCNN Dropping to **78.61** Car/Mod R40 and **57.12** mAP. On nuScenes, VoxelNeXt and TransFusion-Lidar accepted rows are complete: ASAP-v1 is near ROR on Injection, and the explicitly labeled ASAP-pfilter cascade beats ROR on Perturbation for both detectors.

Current contribution target:

1. Adaptive Spherical Purification Units that match LiDAR range-dependent density.
2. Geometry-first anomaly-guided SPU selection with calibrated closed-form features.
3. Selective VP-SDE purification with a trained local point-cloud score network, evaluated without detector retraining.

## Target venue and format

- **Venue**: IEEE ICASSP 2026 (regular paper).
- **Format**: IEEE two-column conference template.
- **Page budget**: typically 4 content pages + 1 page for references (subject to the final ICASSP 2026 call).

## File layout

| File | Section | Target rough length on the final paper |
|------|---------|----------------------------------------|
| `00_abstract.md`     | Abstract                             | 150 - 200 words           |
| `01_introduction.md` | 1. Introduction                      | about 3/4 of column 1     |
| `02_related_work.md` | 2. Related Work                      | about 1/2 column          |
| `03_method.md`       | 3. ASAP Framework                    | 1.5 - 2 columns + figure  |
| `04_experiments.md`  | 4. Experiments                       | 1 - 1.5 columns           |
| `05_conclusion.md`   | 5. Conclusion                        | about 1/4 column          |
| `fig_asap_pipeline.tex` | Figure 1 TikZ source              | LaTeX-ready figure        |
| `tables_icassp.tex`  | Tables 1-2 LaTeX source              | LaTeX-ready tables        |
| `main_icassp.tex`    | First compressed IEEE-style manuscript assembly | LaTeX entry point |
| `detector_selection.md` | Detector selection notes          | internal planning note    |
| `references.bib`     | References (BibTeX)                  | grows as we cite          |

> Experiment records (E0..E6) used to live in `experiments_overview.html` in this folder. They have been moved one level up to `/root/autodl-tmp/ASAP/docs/experiments/` (one Markdown file per experiment plus a master `README.md`). See `docs/README.md` for the top-level navigation.

## Section status

| Section            | Status        | Notes                                      |
|--------------------|---------------|--------------------------------------------|
| Abstract           | revised       | updated with verified KITTI/nuScenes detector roster and pfilter numbers |
| Introduction       | revised       | contribution claim updated to match verified results |
| Related work       | first draft   | 2.1 attacks / 2.2 defenses / 2.3 diffusion purification     |
| Method             | consolidated  | compressed ICASSP-length Markdown with M1/M2/M3 equations, Algorithm 1, and pfilter boundary |
| Experiments        | consolidated  | compressed to main table, claim-critical ablations, pfilter caveat, and runtime caveat |
| LaTeX assembly     | first draft   | `main_icassp.tex` created; integrity check passes; PDF compile not run because no TeX toolchain is installed locally |
| Detector selection | planning note | final roster: KITTI (PointPillars, PV-RCNN) + nuScenes (VoxelNeXt, TransFusion-Lidar); see file for the Waymo → nuScenes decision (2026-05-21) |
| Experiment records | moved out     | now under `/root/autodl-tmp/ASAP/docs/experiments/` (one Markdown file per E<id>) |
| Conclusion         | revised       | numerical claims aligned with current Table 1 |
| References         | revised       | cited keys are present in `references.bib`; placeholder notes removed |

Reviewer pass status on 2026-06-02: Markdown-only reviewer pass completed in `docs/13_markdown_reviewer_pass.md`. Main remaining issue before LaTeX is table layout, not missing experiments.

## Writing conventions

- Write all paper content in **English**, since the final manuscript will be English.
- Do not add HTML comments to submission-facing sections (`00_abstract.md` through `05_conclusion.md`).
- Method names are bold and capitalized in their first appearance, e.g. **Adaptive Radius**, **Anomaly Scorer**, **Selective Diffusion**, and abbreviated as M1, M2, M3 thereafter.
- Equations and symbols stay consistent across sections; symbols are defined once in `03_method.md` and reused.
- Numerical placeholders must not appear in submission-facing sections. If a result is missing, remove the claim or mark it as excluded/diagnostic with a concrete reason.
- All citations use BibTeX keys in `references.bib`; in Markdown drafts, cite as `[@<key>]`.
- Keep each section under its rough length budget so that the final paper still fits 4 columns.

## How to update

- Edit only one section per commit when possible, so reviewers can read the diff cleanly.
- When numerical results land, replace the relevant table/paragraph in `00_abstract.md`, `04_experiments.md`, `05_conclusion.md`, and `Memory/project_status_memory.md` together.
- Keep the status table in this README in sync when a section moves from `skeleton` to `first draft`, `revised`, or `final`.
