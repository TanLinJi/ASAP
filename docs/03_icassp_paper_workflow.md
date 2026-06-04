# ASAP ICASSP paper workflow

This document is the execution workflow for turning ASAP from the current idea and prototype into an ICASSP-ready paper. It is intentionally project-specific: every stage names the ASAP artefacts to read, update, or produce.

Working title: **ASAP: Adaptive Spherical Anomaly-Guided Purification Against LiDAR Point Cloud Attacks**

Target: **ICASSP 2026** regular paper. Use the standard ICASSP-style short conference budget as the working constraint: 4 content pages plus references unless the official author kit says otherwise. Before final formatting, verify the current official author kit and deadline.

## 0. Current entry point

ASAP is not starting from zero. The current state is:

| Area | Status | Evidence |
|------|--------|----------|
| Idea and framing | started | `docs/01_SA-SPD_proposal.html`, `docs/paper/01_introduction.md`, `docs/paper/03_method.md` |
| Environment and data | mostly done | `docs/experiments/E0_environment/`, local `data/`, `checkpoints/`, `third_party/OpenPCDet/` |
| Clean detector baselines | done | `docs/experiments/E1_clean_baselines/` |
| KITTI PointPillars attack-defense loop | done enough for diagnosis | `docs/experiments/E4_main_results/E4.1_kitti_pointpillars.md`, `outputs/*_eval_summary.json` |
| Final ASAP method | revised KITTI candidate | Trained VP-SDE score-net path is implemented; revised Track A keeps learned M3 for Injection/Perturbation and skips M3 for Dropping |
| Multi-detector robustness claim | partial | E4.2 PV-RCNN diagnostic done; E4.3/E4.4 nuScenes accepted no-defense gates complete; nuScenes defense matrix pending |
| nuScenes attack loop | gate complete | Full keyframe-only E2.x attacked sets generated; accepted attacks are Injection original budget and Perturbation `epsilon=0.5`; Dropping is paused after failing the keyframe-only gate |
| Paper draft | first draft only | `docs/paper/`, with numerical placeholders and unverified references |

Immediate rule: do not write stronger claims than the verified experimental matrix supports. If the final method remains local-mean or hybrid rather than a trained diffusion model, rename the paper claims accordingly before submission.

## 1. Stage R0 - freeze the claim

Purpose: decide the exact paper claim before spending more GPU time.

Decision as of 2026-05-29: **Track A is selected.** The paper will pursue the full ASAP diffusion claim, so M3 must include a trained point-cloud score network and the final ASAP row must not rely on the local-mean fallback. The existing local-mean, drop, and hybrid results remain useful as implementation milestones and ablations, but they are not sufficient for the final ICASSP main claim.

Tasks:

1. Read `docs/paper/00_abstract.md`, `01_introduction.md`, `03_method.md`, and `04_experiments.md`.
2. Read `docs/experiments/E4_main_results/E4.1_kitti_pointpillars.md`.
3. Choose one of two paper tracks:
   - **Track A: full ASAP diffusion paper.** Requires implementing and validating a trained point-cloud score network for M3. **Selected.**
   - **Track B: inference-only geometric purification paper.** Uses M1/M2 plus local-mean or hybrid purification; method name and claims must stop implying a trained diffusion model.
4. Freeze the contribution list to 3 bullets maximum.

Gate to pass:

- A one-paragraph thesis statement exists.
- The thesis is supported by an experiment matrix that can realistically finish.
- The method description matches the actual code.

Deliverable:

- Update `docs/paper/README.md` with the chosen track and contribution list.

Track A implementation gate:

- `VPSDEPurifier` loads a trained score network from a checkpoint, not `None`.
- The reverse VP-SDE path is executed and recorded in metadata as a score-net method, not `local_mean_fallback`.
- A smoke model can be trained and used on at least one KITTI attacked frame.
- The final full-val E4.1 rerun shows whether score-net M3 improves over local-mean fallback; if it does not, either revise the method or downgrade the claim before paper freeze.

Initial Track A implementation status:

- Added local PointNet-style score network: `src/asap/diffusion/score_net.py`.
- Added clean-SPU VP-SDE training script: `scripts/train_vpsde_score_net.py`.
- Added score-net ASAP runner: `scripts/run_asap_vpsde_score_net.sh`.
- `python -m asap.pipeline --purifier vp_sde` now accepts `--score_net_ckpt`.
- Smoke check on 2026-05-29:
  - trained `checkpoints/kitti/asap_score_net_smoke.pth` on 1 KITTI frame / 4 patches / 1 epoch / CPU;
  - ran one E2.2 attacked frame through `vp_sde_score_net`;
  - metadata reported `method = vp_sde_score_net`, `n_score_edited_points = 27682`.

The smoke model is not a publishable checkpoint. It only proves that Track A's train-load-purify control path is wired.

## 2. Stage R1 - literature and related-work audit

Purpose: make sure the idea is novel enough and citations are correct.

Tasks:

1. Verify every BibTeX entry in `docs/paper/references.bib`; remove `UNVERIFIED placeholder` notes only after checking metadata.
2. Build a compact related-work matrix with columns:
   - Method
   - Attack/defense type
   - Detector-agnostic?
   - Inference-only?
   - Needs retraining?
   - Local vs scene-wide?
   - Reported dataset/detector
   - Gap that ASAP addresses
3. Cover at minimum:
   - LiDAR point injection, perturbation, dropping attacks.
   - Classical SOR/ROR style denoising.
   - LiDAR-SPD / spherical projection and diffusion.
   - Ada3Diff and point-cloud diffusion purification.
   - Detector families used here: PointPillars, PV-RCNN, VoxelNeXt, TransFusion-Lidar.

Gate to pass:

- No citation used in a claim is unverified.
- Related work does not overstate ASAP's novelty.
- Any copied wording from notes or papers is removed or quoted only in compliant short form.

Deliverables:

- `docs/paper/02_related_work.md` revised.
- `docs/paper/references.bib` verified.
- Optional internal table under `docs/references/related_work_matrix.md`.

## 3. Stage E0 - reproducibility baseline

Purpose: make the current environment and clean baselines defensible.

Tasks:

1. Verify KITTI counts and info files:
   - training: 7481 velodyne/calib/label/image
   - testing: 7518 velodyne/calib/image
   - `kitti_infos_{train,val,trainval,test}.pkl`
   - `kitti_dbinfos_train.pkl`
2. Verify nuScenes trainval infos and checkpoints.
3. Ensure `third_party/README.md` records OpenPCDet upstream, license, and local modifications.
4. Ensure the E0 and E1 experiment records match actual logs.

Gate to pass:

- E0/E1 docs contain exact command, checkpoint path, log path, metric, and known environment patches.
- Clean baselines match the OpenPCDet reference or checkpoint-encoded metric within tolerance.

Deliverables:

- Updated E0/E1 experiment records.
- A short "clean baseline table" for `docs/paper/04_experiments.md`.

## 4. Stage E1 - attack protocol hardening

Purpose: make the attack setup publishable.

Current issue: `src/asap/attacks/` implements detector-agnostic geometric/random variants. That is useful for a first closed loop, but it is weaker than the planned gradient/saliency protocols in the E2 skeletons.

Tasks:

1. Decide whether the paper will use:
   - geometric detector-agnostic attacks only, clearly named as such; or
   - stronger detector-aware attacks for final main results.
2. Update E2.1/E2.2/E2.3 docs to reflect the actual protocol.
3. For each attack, record:
   - threat model
   - point budget or epsilon/drop ratio
   - random seed
   - attacked split size
   - no-defense AP/mAP and ASR
4. If stronger attacks are required, implement or adapt them before running more defenses.

Gate to pass:

- The attack description in the paper exactly matches code and outputs.
- Attack strength is meaningful enough: no-defense should produce a clear metric drop for the class/metric used in the main table.
- No "classic attack reproduction" claim is made for a simplified heuristic attack.

Deliverables:

- Updated `docs/experiments/E2_attacks/*.md`.
- Final attacked datasets under `outputs/attacks/`.
- No-defense evaluation summaries.

## 5. Stage M - method implementation lock

Purpose: lock the method variant before final experiments.

Tasks:

1. Audit M1:
   - `src/asap/geometry/adaptive_spu.py`
   - defaults: `k=16`, `alpha=2.0`, `beta=0.67`, `r_min=0.10`, `r_max=0.40`, `eta=0.30`
2. Audit M2:
   - `src/asap/scoring/anomaly_scorer.py`
   - four features, calibration, tau, attack_kind inference
3. Audit M3:
   - `DropPurifier`
   - `VPSDEPurifier` local-mean fallback
   - `HybridPurifier`
4. Choose the final "ASAP" row:
   - current KITTI two-detector candidate: `ASAP-vpsde-score-net-trackA-revised-dropping-skip`
   - policy rule: b075-ifilter for E2.1 Injection, b075 for E2.2 Perturbation, skip M3 for E2.3 Dropping
   - keep `drop`, local-mean `vp_sde`, `hybrid`, `hybrid-cond`, pure score-net, and raw b075 as ablations
5. Before final paper freeze, decide whether the Dropping skip should be framed as conservative damage avoidance or replaced with a true density-drop reconstruction branch.

Gate to pass:

- Final method has one reproducible CLI command.
- Paper equations and algorithm match the code path actually used.
- Ablations are clearly separated from the final method.

Deliverables:

- Method lock note in `docs/experiments/E4_main_results/E4.1_kitti_pointpillars.md`.
- Updated `docs/paper/03_method.md`.
- Final run scripts under `scripts/`.

## 6. Stage E2 - main experiment matrix

Purpose: fill the paper's main results with a matrix that is large enough but still feasible.

Minimum ICASSP matrix:

| Dataset | Detector | Required status |
|---------|----------|-----------------|
| KITTI | PointPillars | already has full E4.1 diagnostic matrix |
| KITTI | PV-RCNN | needed for detector-agnostic KITTI claim |
| nuScenes | VoxelNeXt | needed for modern 360-degree detector claim |
| nuScenes | TransFusion-Lidar | needed for second modern detector family |

Recommended columns:

| Column | Meaning |
|--------|---------|
| Clean | clean detector reference |
| No defense | attacked input, no purifier |
| Best classical | max of SOR/ROR for that row |
| ASAP | final locked ASAP method |
| ASR reduction | relative to no-defense |

Tasks:

1. Finish E4.2 on KITTI PV-RCNN for the same E2 attacks.
2. Use the current nuScenes keyframe-only attack sets as a low-cost attack-strength gate:
   - outputs: `outputs/attacks/nuscenes_keyframe/E2.{1,2,3}_*/samples/LIDAR_TOP/`
   - protocol caveat: current keyframes edited; historical sweeps remain clean under `MAX_SWEEPS=10`.
3. First run VoxelNeXt no-defense on E2.1/E2.2/E2.3 and inspect mAP/NDS drops. **Done:** Injection 21.97/41.87, Perturbation 51.87/59.95, Dropping 58.91/65.76 against clean 60.52/66.64.
4. Because original Dropping was weak and original Perturbation was only light-to-moderate, revise the nuScenes budget before defense-matrix expansion. **Done:** Perturbation `epsilon=0.5` gives 40.27/51.46, mAP ASR 33.46%; Dropping `drop_ratio=1.0` gives 55.58/63.91, mAP ASR 8.17%.
5. Continue defense evaluation only for accepted nuScenes attacks: Injection original budget and Perturbation `epsilon=0.5`. Keep Dropping paused unless a sweep-aware/saliency-aware protocol is implemented.
6. Parse every OpenPCDet log into JSON summaries using `scripts/parse_eval_summary.py` or `scripts/parse_nuscenes_eval_summary.py`.

Gate to pass:

- Every number in the main table has a log path and a parser output.
- No missing `[XX.X]` remains in the main-result table.
- The best non-ASAP baseline is selected by a clear rule.

Deliverables:

- Updated E4.1-E4.4 experiment records.
- Main table inserted into `docs/paper/04_experiments.md`.

## 7. Stage E3 - ablations and cost

Purpose: support the method design without overloading the paper.

Required ablations for a 4-page paper:

1. **M3 variant ablation:** drop vs local-mean vs hybrid-cond.
2. **M2 selection evidence:** flagged SPU ratio and tau/Youden-J stats.
3. **Cost:** purifier latency and detector latency on the same machine.

Optional ablations if time allows:

1. M1 fixed radius vs adaptive radius.
2. M2 single-feature ablation.
3. Tau sweep.

Tasks:

1. Convert the existing hybrid and vpsde outputs into a concise ablation table.
2. Populate E5.3 first, then E5.1/E5.2/E5.4 only if needed.
3. Populate E6.1/E6.2 from logs and metadata JSONL.

Gate to pass:

- Ablation table explains why the final method was chosen.
- Cost table reports both added purification cost and end-to-end impact.
- Any negative ablation result is discussed honestly.

Deliverables:

- Updated `docs/experiments/E5_ablations/` and `E6_cost/`.
- Tables 2 and 3 in `docs/paper/04_experiments.md`.

## 8. Stage W - paper drafting

Purpose: produce a tight ICASSP paper from verified material.

Writing order:

1. `03_method.md`: lock notation and algorithm first.
2. `04_experiments.md`: fill tables and factual result paragraphs.
3. `01_introduction.md`: rewrite contributions to match results.
4. `02_related_work.md`: sharpen contrast without overclaiming.
5. `00_abstract.md`: write last, with final numbers.
6. `05_conclusion.md`: keep short and avoid unsupported future promises.

Page-budget plan:

| Section | Target |
|---------|--------|
| Abstract | 150-180 words |
| Introduction | 0.75 column |
| Related work | 0.5 column |
| Method | 1.5 columns |
| Experiments | 1.25-1.5 columns |
| Conclusion | 0.25 column |

Gate to pass:

- No `[XX.X]`, `TODO`, or unverified citation notes remain in paper-facing files.
- Every numeric claim links back to one experiment record.
- Claims are phrased no stronger than the completed experiment matrix.

Deliverables:

- Revised `docs/paper/*.md`.
- One assembled manuscript draft, preferably `docs/paper/asap_icassp2026_draft.md` or LaTeX equivalent.

## 9. Stage I - integrity verification

Purpose: catch bad citations, bad numbers, and claim drift before review.

Tasks:

1. Build a claim ledger:
   - claim text
   - section
   - evidence file
   - log/result source
   - status
2. Verify each citation:
   - title, authors, venue, year
   - BibTeX key
   - claim supported by cited source
3. Verify each number:
   - trace paper number -> experiment record -> parser output -> raw log
4. Verify code/result consistency:
   - final method row uses the locked method
   - no stale table values from earlier prototypes

Gate to pass:

- Zero untraceable paper numbers.
- Zero unverified references.
- Zero method/result mismatches.

Deliverables:

- `docs/paper/integrity_check.md`.
- Corrected manuscript.

## 10. Stage V - internal review and revision

Purpose: simulate reviewer objections before submission.

Review personas:

1. Signal processing / ICASSP fit reviewer.
2. 3D detection reviewer.
3. Adversarial robustness reviewer.
4. Reproducibility reviewer.
5. Devil's advocate reviewer.

Review checklist:

1. Is the novelty real relative to LiDAR-SPD/Ada3Diff/SOR/ROR?
2. Are attacks strong enough?
3. Is detector-agnostic evidence sufficient?
4. Is the method actually diffusion-based if M3 is local-mean fallback?
5. Are negative results handled honestly?
6. Are experiments reproducible?

Gate to pass:

- Major objections have either a fix or an explicit scope limitation.
- The paper title/abstract/contributions match the actual evidence.

Deliverables:

- `docs/paper/internal_review.md`.
- Revision plan.
- Revised manuscript.

## 11. Stage F - final formatting and submission package

Purpose: create a submission-ready package.

Tasks:

1. Verify current ICASSP official template, page limit, anonymity, and deadline.
2. Convert Markdown sections to LaTeX or write directly in IEEE template.
3. Create compact tables and one method figure.
4. Build PDF.
5. Run final checks:
   - page count
   - references
   - anonymous if required
   - no local paths in final manuscript
   - no hidden comments/TODOs
   - all figures readable in grayscale

Gate to pass:

- PDF builds without warnings that affect references, figures, or citations.
- Final manuscript contains only verified claims.

Deliverables:

- LaTeX source under `docs/paper/latex/` or agreed final location.
- Submission PDF.
- Optional reproducibility appendix / README.

## 12. Step-by-step execution order from today

Execute in this order:

1. **S1: synchronize status docs.** Update E2/E3/E4/E5/E6 status boards to match actual outputs.
2. **S2: decide Track A vs Track B.** This determines whether to invest in a trained VP-SDE score net.
3. **S3: lock final method.** Pick final ASAP row and rename claims if needed.
4. **S4: harden E2 attack records.** Document actual geometric attack protocols or implement stronger attacks.
5. **S5: finish KITTI PV-RCNN E4.2.** This is the lowest-cost detector-agnostic extension.
6. **S6: decide nuScenes robustness scope.** Use VoxelNeXt no-defense on the keyframe-only attacks as the gate; only expand E4.3/E4.4 if mAP/NDS drops are meaningful.
7. **S7: fill ablation and cost records.** Prioritize M3 ablation and latency.
8. **S8: rewrite experiments section.** Fill main table and remove placeholders.
9. **S9: update method/introduction/abstract.** Make claims match results.
10. **S10: run integrity check.** Verify all citations and numbers.
11. **S11: internal review.** Fix high-risk reviewer objections.
12. **S12: format final submission.**

## 13. Current next action

S1 and S2 are complete enough to proceed with Track A. The first learned-M3 validation loop now has a KITTI/PointPillars policy candidate and a PV-RCNN detector-extension diagnostic:

- Score-net v1 checkpoint: `checkpoints/kitti/asap_score_net_trackA_v1.pth`.
- Training log: `outputs/asap_vpsde_score_net/logs/train_score_net_trackA_v1.log`.
- Control subset: `outputs/smoke/score_net_v1_subset_20260529_2224/`; metadata confirms `method = vp_sde_score_net`.
- Completed full purification: E2.2 perturbation -> `outputs/asap_vpsde_score_net/kitti/E2.2_perturbation/velodyne_purified/`.
- Completed PointPillars eval: `E4_asap_vpsde_score_net_E2.2_pp`, Car/Moderate 3D AP_R40 = **58.38**, mAP3D_R40 Moderate = **29.41**.
- Comparison: score-net v1 improves no-defense (56.54 / 27.32) but trails local-mean `ASAP-vpsde` (60.10 / 32.79), so it is not final.
- Completed anchor follow-up: `score_anchor_blend=0.75`, tags `E4_asap_vpsde_score_net_b075_E2.{1,2,3}_pp`.
  - E2.1 Injection: Car/Moderate 3D AP_R40 = **59.97**, mAP3D_R40 Moderate = **39.63**.
  - E2.2 Perturbation: Car/Moderate 3D AP_R40 = **61.04**, mAP3D_R40 Moderate = **34.69**.
  - E2.3 Dropping: Car/Moderate 3D AP_R40 = **68.49**, mAP3D_R40 Moderate = **35.41**.
- Comparison: score-net+anchor improves local-mean on E2.2 (+0.94 Car AP / +1.90 mAP) and E2.3 (+1.81 Car AP / +2.76 mAP), but raw b075 does not solve E2.1 Injection.
- Completed Injection-filter follow-up: `score_injection_filter_radius=0.4`, `score_injection_filter_min_neighbors=5`, enabled only when `attack_kind == "injection"`.
  - E2.1 Injection b075-ifilter: Car/Moderate 3D AP_R40 = **65.71**, mAP3D_R40 Moderate = **52.18**.
  - Metadata: 3769 rows, `method = vp_sde_score_net`, `score_anchor_blend = 0.75`, `score_injection_filter_enabled = true`, filter drop rate **4.98%**.
  - Non-injection smoke: `outputs/smoke/b075_ifilter_disabled_non_injection_20260530_104312/summary.json` confirms E2.2/E2.3 run with the same filter parameters but `score_injection_filter_enabled = false`, zero filter drops, and input/output point counts unchanged.
- Combined policy summary: `outputs/asap_vpsde_score_net_trackA_policy/kitti_pointpillars_eval_summary.json`.
- PV-RCNN E4.2 diagnostic summary: `outputs/asap_vpsde_score_net_trackA_policy/kitti_pvrcnn_eval_summary.json`.
  - E2.1 Injection: no-defense **63.16** Car / **48.22** mAP; Track A **72.59** / **57.73**.
  - E2.2 Perturbation: no-defense **3.50** / **2.45**; Track A **10.96** / **5.98**.
  - E2.3 Dropping: no-defense **78.61** / **57.12**; Track A **76.15** / **42.45**.
- Revised policy summary: `outputs/asap_vpsde_score_net_trackA_revised_dropping_skip/kitti_policy_eval_summary.json`.
  - Rule: E2.1 b075-ifilter, E2.2 b075, E2.3 skip M3 / no purification.
  - PointPillars: **65.71 / 52.18**, **61.04 / 34.69**, **71.22 / 48.42** for E2.1/E2.2/E2.3.
  - PV-RCNN: **72.59 / 57.73**, **10.96 / 5.98**, **78.61 / 57.12** for E2.1/E2.2/E2.3.
- PV-RCNN SOR/ROR baselines: `outputs/asap_vpsde_score_net_trackA_revised_dropping_skip/kitti_pvrcnn_sor_ror_eval_summary.json`.
  - SOR: **66.82 / 52.36**, **0.89 / 0.91**, **49.38 / 38.87** for E2.1/E2.2/E2.3.
  - ROR: **72.77 / 57.77**, **1.94 / 1.17**, **59.05 / 43.12** for E2.1/E2.2/E2.3.
  - Interpretation: ROR remains marginally best on PV-RCNN Injection; ASAP is strongest on Perturbation and Dropping.
- Execution policy update: long runs should use `tmux`; where feasible use both T4 GPUs. `scripts/eval_attacked_kitti.sh` supports `NUM_GPUS=2`, and `scripts/run_asap_vpsde_score_net_dual_gpu.sh` shards score-net purification across GPUs.

Next gate: run TransFusion-Lidar no-defense on accepted nuScenes attacks only.

Current nuScenes state:

- Added keyframe-only attack module: `src/asap/attacks/nuscenes_keyframe_attack.py`.
- Added attack runner: `scripts/run_nuscenes_keyframe_attacks.sh`.
- Added safe eval swap script: `scripts/eval_attacked_nuscenes.sh`.
- Added no-defense runner: `scripts/eval_nuscenes_keyframe_no_defense.sh`.
- Added parser: `scripts/parse_nuscenes_eval_summary.py`.
- Full attacked outputs:
  - E2.1 Injection: 6019 keyframes, 158253 targeted objects, 7912650 injected points.
  - E2.2 Perturbation: 6019 keyframes, 158253 targeted objects, 11973287 perturbed object points.
  - E2.3 Dropping: 6019 keyframes, 158253 targeted objects, 5953034 dropped points.
- Completed VoxelNeXt no-defense gate: Injection 21.97/41.87 (63.70% mAP ASR), Perturbation 51.87/59.95 (14.29% mAP ASR), Dropping 58.91/65.76 (2.66% mAP ASR).
- Gate interpretation: Injection is strong; original Perturbation is light-to-moderate; Dropping is too weak under the current keyframe-only `drop_ratio=0.5` protocol.
- Completed budget gate: Perturbation `epsilon=0.5` gives 40.27/51.46 (33.46% mAP ASR); Dropping `drop_ratio=1.0` gives 55.58/63.91 (8.17% mAP ASR).
- Completed TransFusion-Lidar accepted no-defense gate: Injection gives 19.13/39.45 (70.37% mAP ASR); Perturbation `epsilon=0.5` gives 43.02/52.99 (33.37% mAP ASR).
- Decision rule: accept Injection original budget and Perturbation `epsilon=0.5` for nuScenes detector-transfer and defense evaluation; keep Dropping paused as a weak diagnostic unless replaced by a sweep-aware/saliency-aware protocol.
