# ASAP experiments — master index

This directory hosts every per-experiment record for the ASAP project. The intent is that **every numerical claim in the paper can be traced back to exactly one Markdown file here**, which in turn cites the exact command, log, checkpoint and analysis behind that number.

## Conventions

- One sub-folder per experiment family (`E0_environment`, `E1_clean_baselines`, ...).
- One Markdown file per experiment (`E0.1_kitti_data_preparation.md`, `E4.1_kitti_pointpillars.md`, ...).
- Every experiment file is built from `_template.md` and must keep its sections 1-10 intact.
- IDs follow `E<family>.<index>`; once allocated, an ID is **never re-used** even if the experiment is deprecated.
- Paper-side claims should cite the experiment by ID, e.g. *(see E4.1)*; reverse-link to the paper from each experiment file's *Linked paper section* row.

## Method module legend

| Module | Name                | Code path                            | Defined in |
|--------|---------------------|--------------------------------------|------------|
| `M1`   | Adaptive Radius     | `src/asap/geometry/`                 | `paper/03_method.md` §3.3 |
| `M2`   | Anomaly Scorer      | `src/asap/scoring/`                  | `paper/03_method.md` §3.4 |
| `M3`   | Selective Diffusion | `src/asap/diffusion/`, `src/asap/pipeline/` | `paper/03_method.md` §3.5 |

## Glossary of acronyms

| Acronym | Expansion | In ASAP |
|---------|-----------|---------|
| `SPU`   | Spherical Purification Unit | Local ball used by M1/M2/M3. |
| `KNN`   | k-Nearest Neighbors         | Local scale estimator behind M1. |
| `PCA`   | Principal Component Analysis | Anisotropy feature in M2. |
| `vMF`   | von Mises-Fisher distribution | Angular concentration feature in M2. |
| `VP-SDE`| Variance-Preserving Stochastic Differential Equation | Diffusion process used in M3. |
| `ROC`   | Receiver Operating Characteristic | Calibration curve for the M2 threshold $\tau$. |
| `Youden-J` | Youden index $J = \mathrm{TPR} - \mathrm{FPR}$ | Picks the optimal $\tau$ from the ROC curve. |
| `AP / mAP` | (mean) Average Precision | Detection metric on KITTI. |
| `NDS`   | nuScenes Detection Score    | Composite metric on nuScenes (`E1.3 / E1.4 / E4.3 / E4.4`). |
| `mAPH` | mean AP weighted by Heading | Waymo metric; only relevant to the deferred `E4.5_deferred_waymo_mppnet.md`. |
| `ASR`   | Attack Success Rate         | Primary defense metric for ASAP. |
| `SOR / ROR` | Statistical / Radius Outlier Removal | Classical denoising baselines. |

## Experiment families

| Family | Folder | What lives here |
|--------|--------|-----------------|
| `E0` | [`E0_environment/`](E0_environment/README.md)              | Data, environment, checkpoints, infos. |
| `E1` | [`E1_clean_baselines/`](E1_clean_baselines/README.md)       | Clean-scene detection numbers, one per (dataset, detector). |
| `E2` | [`E2_attacks/`](E2_attacks/README.md)                       | Adversarial attack reproductions; produce perturbed val sets. |
| `E3` | [`E3_baseline_defenses/`](E3_baseline_defenses/README.md)   | Non-ASAP defense baselines (SOR/ROR/uniform diffusion/external). |
| `E4` | [`E4_main_results/`](E4_main_results/README.md)             | ASAP main detector-agnostic results; back the paper main table. |
| `E5` | [`E5_ablations/`](E5_ablations/README.md)                   | Module-level ablations of M1/M2/M3 and threshold $\tau$. |
| `E6` | [`E6_cost/`](E6_cost/README.md)                             | Wall-clock cost and selective ratio measurements. |
| `E7` | [`E7_loss_functions/`](E7_loss_functions/README.md)          | Score-net loss-function upgrades beyond the Track A baseline. |

## Status board

Legend: `done` = finished with results in file, `running` = currently executing, `pending` = not started, `blocked` = waiting on upstream, `optional` = nice-to-have.

### E0 — Environment

| ID | Name | Status | Notes |
|----|------|--------|-------|
| E0.1 | KITTI 3D object detection data | **done**    | train 3712 / val 3769 / test 7518, verified. |
| E0.2 | `asap` conda environment       | **done**    | Python 3.9 + PyTorch 2.5.1+cu124 + spconv 2.3.6. |
| E0.3 | OpenPCDet backend + KITTI infos | **done**   | 4 info pkls + 19700 gt_database bins. |
| E0.4 | Detector pretrained checkpoints | **done** | All 4 checkpoints end-to-end verified bit-exact: PointPillars 77.28 (E1.1), PV-RCNN 83.69 (E1.2), VoxelNeXt mAP 60.52 / NDS 66.64 (E1.3), TransFusion-L mAP 64.57 / NDS 69.43 (E1.4). |
| E0.5 | nuScenes data preparation      | **done** | `v1.0-trainval` downloaded (293 GiB / 11 archives, 17:49), extracted (92 min via pigz), `create_nuscenes_infos` finished 2026-05-22 00:16 (3 h 26 min). 850 scene / 34149 sample / 1.17M annotation. Canonical 700-train / 150-val / 28130-train-sample / 6019-val-sample split. |

### E1 — Clean baselines

| ID | Dataset      | Detector           | Metric              | Status |
|----|--------------|--------------------|---------------------|--------|
| E1.1 | KITTI val    | PointPillars       | 3D AP (Car/Ped/Cyc) | **done** — 77.28 / 52.29 / 62.61 vs OpenPCDet 77.28 / 52.29 / 62.68 |
| E1.2 | KITTI val    | PV-RCNN            | 3D AP (Car/Ped/Cyc) | **done** — 83.69 / 54.84 / 69.48 (R11 Mod), bit-exact match to ckpt filename `pv_rcnn_8369.pth` |
| E1.3 | nuScenes val | VoxelNeXt (0.075)  | mAP / NDS           | **done** — v1.0-trainval val (6019 samples) mAP **0.6052** / NDS **0.6664**; Δ vs OpenPCDet README -0.0001 / -0.0001 (bit-exact). |
| E1.4 | nuScenes val | TransFusion-Lidar  | mAP / NDS           | **done** — v1.0-trainval val (6019 samples) mAP **0.6457** / NDS **0.6943**; Δ vs OpenPCDet README -0.0001 / **0.0000** (NDS bit-exact). |

### E2 — Adversarial attacks

| ID | Attack | Family | Status |
|----|--------|--------|--------|
| E2.1 | Point Injection     | injection    | **done for KITTI/PointPillars geometric attack** — 3769 attacked val bins, no-defense Car/Mod R40 = 60.10; stronger detector-aware variant still optional. |
| E2.2 | Point Perturbation  | perturbation | **done for KITTI/PointPillars geometric attack** — 3769 attacked val bins, no-defense Car/Mod R40 = 56.54; stronger PGD-style variant still optional. |
| E2.3 | Point Dropping      | dropping     | **done for KITTI/PointPillars random object-point dropping** — 3769 attacked val bins, no-defense Car/Mod R40 = 71.22; saliency-based variant still optional. |

### E3 — Baseline defenses

| ID | Defense | Type | Status |
|----|---------|------|--------|
| E3.1 | SOR                       | statistical denoising | **done for KITTI/PointPillars E2.1-E2.3** — outputs under `outputs/defenses/kitti/E3.1_sor/`, metrics included in E4.1. |
| E3.2 | ROR                       | radius denoising      | **done for KITTI/PointPillars E2.1-E2.3** — outputs under `outputs/defenses/kitti/E3.2_ror/`, metrics included in E4.1. |
| E3.3 | Uniform SPU diffusion     | ASAP w/o M2 gating    | **done as SPD-style fixed-SPU proxy** — PointPillars / Perturbation 57.16 / 30.35; see E5.3. |
| E3.4 | External published purifier | external reference   | optional |

### E4 — ASAP main results

| ID | Dataset      | Detector           | Attack coverage | Defense coverage | Status |
|----|--------------|--------------------|-----------------|------------------|--------|
| E4.1 | KITTI val    | PointPillars       | E2.1/2.2/2.3 | No-def, SOR, ROR, ASAP-drop, ASAP-vpsde, hybrid ablations, Track A score-net v1, score-net+anchor, Track A policy | **Track A/B attack-conditional candidate** — b075-ifilter fixes E2.1 mAP (65.71 Car / 52.18 mAP), while `t_star=0.08` is current best on E2.2 (61.26 / 34.76); `t_star=0.10` checked and not promoted. |
| E4.2 | KITTI val    | PV-RCNN            | E2.1/2.2/2.3 | No-def, Track A policy, Track B E2.2 | **diagnostic done** — Injection transfers (+9.42 Car AP); `t_star=0.08` improves Perturbation to 11.23 / 6.26; `t_star=0.10` is not cross-detector consistent; Dropping uses skip policy. |
| E4.3 | nuScenes val | VoxelNeXt (0.075)  | accepted E2.1 + E2.2 eps0.5; Dropping excluded | No-def, SOR, ROR, ASAP-v1, ASAP-pfilter diagnostic | **done** — Injection ASAP-v1 near ROR; Perturbation ASAP-pfilter 52.27/61.23 beats ROR. |
| E4.4 | nuScenes val | TransFusion-Lidar  | accepted E2.1 + E2.2 eps0.5; Dropping excluded | No-def, SOR, ROR, ASAP-v1 Injection, ASAP-pfilter Perturbation | **done** — Injection ASAP-v1 near ROR; Perturbation ASAP-pfilter 54.40/61.98 beats ROR. |
| E4.5 | Waymo val    | MPPNet (temporal)  | n/a          | n/a                      | **deferred** — out of ICASSP scope (`E4.5_deferred_waymo_mppnet.md`) |

### E5 — Ablations (KITTI + PointPillars unless noted)

| ID | Target | Description | Status |
|----|--------|-------------|--------|
| E5.1 | M1 adaptive radius   | fixed $r_1 \in \{0.10, 0.20, 0.30, 0.40\}$ vs adaptive rule | **done** — fixed `r=0.40` slightly exceeds adaptive Track A on KITTI Perturbation but edits more points; use as corrected precision/conservatism diagnostic. |
| E5.2 | M2 feature subset    | lightweight feature diagnostic plus detector validation for `only_f_d` / `drop_f_d` on KITTI Perturbation | **done** — `f_d` is dominant on calibration SPUs, but detector-level `only_f_d` nearly matches full Track A; use as conservative diagnostic evidence. |
| E5.3 | M3 selective gating / purifier variants | fixed-SPU all-unit, adaptive-SPU all-unit, drop/local-mean/hybrid variants vs Track A | **done for paper Table 2** |
| E5.4 | Threshold $\tau$     | focused sweep around ROC-calibrated operating point on KITTI Perturbation | **done** — stricter `tau=0.036/0.048` underperforms `tau=0.024`; keep default. |
| E5.5 | Qualitative selectivity | BEV point-cloud comparisons and per-frame edit/drop statistics for representative KITTI and nuScenes rows | **done** — six representative BEV panels and summary generated under `outputs/qualitative/E5.5_selectivity/`. |
| E5.6 | Track B PV-RCNN diagnostics | KITTI/PV-RCNN E2.2 pfilter, blend, and `t_star` diagnostics | **done** — pfilter and blend050 are negative; `t_star=0.08` improves both PV-RCNN and PointPillars; `t_star=0.10` is closed and not promoted. |

### E6 — Cost

| ID | Measurement | Status |
|----|-------------|--------|
| E6.1 | Per-frame inference latency (ms): ASAP preprocessing throughput | **partial, log-derived** — `outputs/cost/E6_summary/` now summarizes completed pipeline logs. |
| E6.2 | Selective ratio $\lvert C^\star\rvert / \lvert C\rvert$ vs attack family | **done for current Track A / pfilter outputs** — `outputs/cost/E6_summary/` reports SPU and point-edit ratios. |

### E7 — Score-net loss functions

| ID | Target | Description | Status |
|----|--------|-------------|--------|
| E7.1 | VP-SDE score-net loss | Time-balanced DSM, local geometry consistency, density preservation, and paired attacked-clean fine-tuning | **planned** — start with L1 `time_sigma2` after git checkpoint. |

## Milestones

| Milestone | Includes | Output | Paper hook |
|-----------|----------|--------|------------|
| **MS1** — KITTI closed loop          | E0.4 (PointPillars) → E1.1 → E2.x → E3.1 → E3.3 → E4.1 + E5.* + E6.* | First end-to-end result with ablation and cost. | Table 1 (KITTI rows), Table 2, Table 3. |
| **MS2** — KITTI detector extension   | E0.4 (PV-RCNN) → E1.2 → E4.2                                          | Detector-agnostic on KITTI.                       | Table 1 (KITTI PV-RCNN row). |
| **MS3** — nuScenes extension         | E0.5 (nuScenes data) → E0.4 (VoxelNeXt-NuScenes / TransFusion-L) → E1.3, E1.4 → E2.x@nuScenes → E4.3, E4.4 | Detector-agnostic on nuScenes with modern SOTA detectors. | Table 1 (nuScenes rows), abstract ASR-reduction number. |
| **MS4** — Paper freeze               | Complete figures, fill all `[XX.X]`, verify references                | ICASSP-ready manuscript.                          | Whole paper + `references.bib`. |

## Dependency overview

```
E0.1 KITTI ─┐
E0.2 env   ─┼─► E0.3 infos ─┬─► E1.1 PointPillars baseline ─┐
E0.4 ckpt  ─┘               ├─► E1.2 PV-RCNN baseline      ─┤
                            └─► E2.1/2.2/2.3 attacks ──────┐│
                                                           ││
E0.5 nuScn ─► E1.3/1.4 baselines ─► E2.x on nuScenes ─────┤│
                                                           ││
                                                  E3.1 SOR ─┤│
                                                  E3.2 ROR ─┤│
                                                  E3.3 Uni ─┤│
                                                            │└─► E4.1 KITTI/PointPillars main
                                                            └──► E4.2 KITTI/PV-RCNN
                                                                 E4.3 nuScenes/VoxelNeXt
                                                                 E4.4 nuScenes/TransFusion-L
                                                                 E4.5 Waymo/MPPNet (deferred)
                                                                       │
                                                                       └─► E5.* ablations, E6.* cost
```

Key invariants:

- E1 must precede the same (dataset, detector) row in E4.
- E2 must produce attacked val sets before E3 / E4 can be run on them.
- E5 ablations stay on KITTI + PointPillars unless explicitly noted, to keep the design space small.
- E6 cost is measured on the same machine that produces E4.1, so latency rows are comparable.

## How to add a new experiment

1. Pick the right family folder; if the family does not exist yet, create it with its own `README.md`.
2. Allocate the next free ID inside that family (`E<family>.<index>`).
3. Add a row in the status board above.
4. Copy [`_template.md`](_template.md) to `E<family>/<id>_<short_name>.md` and fill at least Sections 1 (purpose), 2 (dependencies) and 3 (setup) before running anything.
5. Update the matching paper section so that the new experiment is referenced once it produces a number.
