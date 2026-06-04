# ASAP Project Memory Dump
**Date**: 2026-05-31

## Project basics
- Project root: `/root/autodl-tmp/ASAP/`
- Method: ASAP = Adaptive Spherical Anomaly-Guided Purification
- Paper title: `ASAP: Adaptive Spherical Anomaly-Guided Purification Against LiDAR Point Cloud Attacks`
- Target: ICASSP 2026; detector-agnostic, inference-only LiDAR purification.
- User preference: do not automatically run Git add/commit/push.
- User preference: for long-running code, prefer `tmux` sessions so jobs survive SSH/network disconnects.
- User preference: when GPUs are available, prefer using both T4 cards in parallel where the code path can be safely sharded or distributed.

## Environment / data
- KITTI at `/root/autodl-tmp/ASAP/data/kitti/`; val has 3769 frames.
- OpenPCDet at `/root/autodl-tmp/ASAP/third_party/OpenPCDet`; `data/kitti` symlinked.
- Conda env `asap`: Python 3.9, PyTorch 2.5.1+cu124, spconv 2.3.6.
- OpenPCDet CUDA runs need `LD_LIBRARY_PATH=/root/miniconda3/envs/asap/lib/python3.9/site-packages/torch/lib:/usr/local/cuda-12.4/lib64`.
- nuScenes trainval is at `/root/autodl-tmp/ASAP/data/nuscenes/v1.0-trainval`; OpenPCDet loads 10 sweeps by default (`MAX_SWEEPS=10`).

## ASAP implementation status as of 2026-05-30
- M1 in `src/asap/geometry/adaptive_spu.py`: KD-tree stride-net FPS, ~0.63s/frame.
- M2 in `src/asap/scoring/anomaly_scorer.py`: 4 features (compactness, PCA anisotropy, vMF kappa, density ratio), log1p, logistic, ROC/Youden-J calibration.
- `ScorerConfig` now has backward-compatible `attack_kind: Optional[str]` and `from_json` filters unknown keys.
- `calibrate_from_dirs` infers `attack_kind` from added-vs-dropped point ratios and stores it in `ScorerConfig`.
- M3 in `src/asap/diffusion/selective_purifier.py`:
  - `DropPurifier` (deletion prototype) — kept as ablation only.
  - `VPSDEPurifier` with `_local_mean_smooth` fallback (8-NN attraction over non-flagged anchors) — historical ablation, not the Track A final candidate.
  - `VPSDEPurifier` can load `checkpoints/kitti/asap_score_net_trackA_v1.pth` and run `method = vp_sde_score_net` with `score_anchor_blend=0.75`.
  - `VPSDEPurifier` has an Injection-only score-path radius filter controlled by `score_injection_filter_radius` and `score_injection_filter_min_neighbors`; it enables only when `attack_kind == 'injection'`.
  - `HybridPurifier` combines anisotropy-gated cluster drops + optional isolation-radius drops + local-mean smoothing.
  - `PurifierConfig.attack_kind` controls `HybridPurifier`: isolation drop is enabled only when `attack_kind is None` or `attack_kind == 'injection'`, disabled for `perturbation` and `dropping`.
- CLI: `python -m asap.pipeline` supports `--purifier vp_sde`, `--score_net_ckpt`, `--score_anchor_blend`, Injection-filter knobs, hybrid knobs, multiprocessing `--workers`, and `--attack_kind {injection,perturbation,dropping}` override. If no override, `scorer_cfg.attack_kind` is passed to `purifier_cfg.attack_kind`.

## KITTI PointPillars E4.1 results (clean Car/Mod 3D AP_R40 = 78.40)

Car / Moderate 3D AP_R40 / ASR vs clean:
- E2.1 Injection: no-defense 60.10 / 23.3%; SOR 58.00 / 26.0%; ROR 66.07 / 15.7%; ASAP-drop 59.47 / 24.1%; ASAP-vpsde 60.18 / 23.2%; ASAP-hybrid 59.80 / 23.7%; ASAP-hybrid-co 60.18 / 23.2%; ASAP-hybrid-cond 59.80 / 23.7%; ASAP-vpsde-score-b075 59.97 / 23.5%; ASAP-vpsde-score-b075-ifilter 65.71 / 16.2%.
- E2.2 Perturbation: no-defense 56.54 / 27.9%; SOR 34.12 / 56.5%; ROR 41.65 / 46.9%; ASAP-drop 47.88 / 38.9%; ASAP-vpsde 60.10 / 23.3%; ASAP-hybrid 58.41 / 25.5%; ASAP-hybrid-co 59.85 / 23.7%; ASAP-hybrid-cond 59.85 / 23.7%; ASAP-vpsde-score-b075 61.04 / 22.1% (best ASAP Car).
- E2.3 Dropping: no-defense 71.22 / 9.2%; SOR 40.16 / 48.8%; ROR 49.91 / 36.3%; ASAP-drop 65.89 / 16.0%; ASAP-vpsde 66.68 / 14.9%; ASAP-hybrid 67.67 / 13.7%; ASAP-hybrid-co 67.63 / 13.7%; ASAP-hybrid-cond 67.63 / 13.7%; ASAP-vpsde-score-b075 68.49 / 12.6% (best ASAP Car, still below no-defense).

mAP (Car+Ped+Cyc, Mod, 3D AP_R40):
- E2.1: no-def 38.73; SOR 43.77; ROR 51.30; ASAP-drop 41.62; ASAP-vpsde 41.71; ASAP-hybrid 42.44; ASAP-hybrid-co 41.71; ASAP-hybrid-cond 42.44; ASAP-vpsde-score-b075 39.63; ASAP-vpsde-score-b075-ifilter 52.18 (best).
- E2.2: no-def 27.32; SOR 14.99; ROR 19.46; ASAP-drop 18.96; ASAP-vpsde 32.79; ASAP-hybrid 30.78; ASAP-hybrid-co 32.62; ASAP-hybrid-cond 32.62; ASAP-vpsde-score-b075 34.69 (best).
- E2.3: no-def 48.42 (best); SOR 28.44; ROR 34.20; ASAP-drop 31.57; ASAP-vpsde 32.65; ASAP-hybrid 32.86; ASAP-hybrid-co 32.85; ASAP-hybrid-cond 32.85; ASAP-vpsde-score-b075 35.41 (best ASAP).

## Headline interpretation as of 2026-05-30
- ASAP-drop (deletion) is net-negative and should not be final ASAP evidence.
- ASAP-vpsde/local-mean is now an ablation, not the Track A candidate.
- Full hybrid proves isolation drops can raise injection mAP (42.44 vs vpsde 41.71) but hurts perturbation.
- Cluster-only hybrid confirms the perturbation regression comes from the isolation-radius branch, not the cluster-drop branch.
- Attack-conditional hybrid (`hybrid-cond`) enables isolation only for injection and disables it for perturbation/dropping. It is the best single hybrid ablation: Injection mAP 42.44, Perturbation Car 59.85, Dropping Car 67.63. It still does not beat ROR on Injection Car AP (66.07) and does not beat vpsde-score-b075 on Perturbation/Dropping, so report it as an ablation, not the final ASAP row.
- Track A attack-conditional policy is the current KITTI/PointPillars learned-M3 candidate: E2.1 uses b075-ifilter (65.71 / 52.18), E2.2 uses b075 (61.04 / 34.69), E2.3 uses b075 (68.49 / 35.41). E2.1 Car AP is still 0.36 below ROR, but mAP exceeds ROR by 0.88.
- PV-RCNN E4.2 detector-extension diagnostic is mixed: Track A improves E2.1 Injection (63.16 -> 72.59 Car, 48.22 -> 57.73 mAP) and E2.2 Perturbation (3.50 -> 10.96 Car, 2.45 -> 5.98 mAP), but regresses E2.3 Dropping (78.61 -> 76.15 Car, 57.12 -> 42.45 mAP). This blocks freezing the current three-attack policy as the final detector-agnostic row.

## Outputs added by attack-conditional hybrid
- Code: `src/asap/scoring/anomaly_scorer.py`, `src/asap/pipeline/__init__.py`, `src/asap/diffusion/selective_purifier.py`.
- Scorer configs patched with `attack_kind`: `outputs/asap/configs/scorer_E2.{1,2,3}_*.json`.
- Purified outputs: `outputs/asap_hybrid_conditional/kitti/E2.{1,2,3}_*/velodyne_purified/`.
- Metadata: `outputs/asap_hybrid_conditional/kitti/E2.{1,2,3}_*/meta.jsonl`.
- Summary: `outputs/asap_hybrid_conditional/kitti_pointpillars_eval_summary.json`.
- Scripts: `scripts/run_asap_hybrid_conditional.sh`, `scripts/eval_asap_hybrid_conditional_all.sh`.
- OpenPCDet tags: `E4_asap_hybrid_cond_E2.{1,2,3}_pp`.
- Doc updated: `docs/experiments/E4_main_results/E4.1_kitti_pointpillars.md`.

## Outputs added by Track A score-net+anchor b075 / b075-ifilter
- Checkpoint: `checkpoints/kitti/asap_score_net_trackA_v1.pth`.
- Purified outputs: `outputs/asap_vpsde_score_net_anchor_b075/kitti/E2.{1,2,3}_*/velodyne_purified/`.
- Metadata: `outputs/asap_vpsde_score_net_anchor_b075/kitti/E2.{1,2,3}_*/meta.jsonl`.
- Summary: `outputs/asap_vpsde_score_net_anchor_b075/kitti_pointpillars_eval_summary.json`.
- Injection-filter output: `outputs/asap_vpsde_score_net_anchor_b075_ifilter/kitti/E2.1_injection/velodyne_purified/`.
- Injection-filter metadata: 3769 rows, all `method = vp_sde_score_net`, `score_anchor_blend = 0.75`, `score_injection_filter_enabled = true`; filter drops 22,332,733 / 448,289,577 points (4.98%).
- Injection-filter summary: `outputs/asap_vpsde_score_net_anchor_b075_ifilter/kitti_pointpillars_eval_summary.json`.
- Track A policy summary: `outputs/asap_vpsde_score_net_trackA_policy/kitti_pointpillars_eval_summary.json`.
- PV-RCNN diagnostic summary: `outputs/asap_vpsde_score_net_trackA_policy/kitti_pvrcnn_eval_summary.json`.
- Revised Track A summary: `outputs/asap_vpsde_score_net_trackA_revised_dropping_skip/kitti_policy_eval_summary.json`.
  - Rule: E2.1 b075-ifilter, E2.2 b075, E2.3 skip M3 / no purification.
  - PointPillars E2.1/E2.2/E2.3 Car/mAP: 65.71/52.18, 61.04/34.69, 71.22/48.42.
  - PV-RCNN E2.1/E2.2/E2.3 Car/mAP: 72.59/57.73, 10.96/5.98, 78.61/57.12.
- PV-RCNN SOR/ROR summary: `outputs/asap_vpsde_score_net_trackA_revised_dropping_skip/kitti_pvrcnn_sor_ror_eval_summary.json`.
  - SOR E2.1/E2.2/E2.3 Car/mAP: 66.82/52.36, 0.89/0.91, 49.38/38.87.
  - ROR E2.1/E2.2/E2.3 Car/mAP: 72.77/57.77, 1.94/1.17, 59.05/43.12.
  - ROR is marginally best on PV-RCNN Injection; ASAP is best on PV-RCNN Perturbation and Dropping.
- PV-RCNN run log: `outputs/asap_vpsde_score_net_trackA_policy/logs/asap_e42_pvrcnn_tracka_20260530_105446.log`.
- Non-injection filter-disabled smoke: `outputs/smoke/b075_ifilter_disabled_non_injection_20260530_104312/summary.json` confirms E2.2/E2.3 have `score_injection_filter_enabled = false`, zero filter drops, and unchanged point counts when filter params are present.
- Scripts: `scripts/run_asap_vpsde_score_net.sh`, `scripts/run_asap_vpsde_score_net_dual_gpu.sh`.
- OpenPCDet tags: `E4_asap_vpsde_score_net_b075_E2.{1,2,3}_pp`.
- OpenPCDet Injection-filter tag: `E4_asap_vpsde_score_net_b075_ifilter_E2.1_pp`.
- Helper script: `scripts/build_tracka_policy_summary.py`.

## nuScenes keyframe-only attack/eval status
- Current protocol: keyframe-only geometric/random attacks. The edited current-frame files replace `samples/LIDAR_TOP`; historical sweeps remain clean because OpenPCDet nuScenes configs use `MAX_SWEEPS=10`. Paper text must report this caveat.
- Added attack code: `src/asap/attacks/nuscenes_keyframe_attack.py`.
- Added attack runner: `scripts/run_nuscenes_keyframe_attacks.sh`.
- Added safe eval script: `scripts/eval_attacked_nuscenes.sh`; it file-locks the nuScenes swap and restores `samples/LIDAR_TOP` with `trap`.
- Added no-defense eval orchestrator: `scripts/eval_nuscenes_keyframe_no_defense.sh`.
- Added accepted-attack eval orchestrator: `scripts/eval_nuscenes_accepted_no_defense.sh`.
- Added metrics parser: `scripts/parse_nuscenes_eval_summary.py`.
- Full nuScenes attacked outputs under `outputs/attacks/nuscenes_keyframe/`:
  - E2.1 Injection: 6019 keyframes, 158253 targeted objects, 7912650 injected points.
  - E2.2 Perturbation: 6019 keyframes, 158253 targeted objects, 11973287 perturbed object points.
  - E2.3 Dropping: 6019 keyframes, 158253 targeted objects, 5953034 dropped points; original keyframe points 208978400, remaining 203025366.
- VoxelNeXt no-defense eval completed in tmux session `asap_nusc_voxelnext_nodef_20260530_155704`.
  - Command: `NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 DETECTOR=voxelnext bash scripts/eval_nuscenes_keyframe_no_defense.sh`.
  - Log: `outputs/attacks/nuscenes_keyframe/asap_nusc_voxelnext_nodef_20260530_155704.log`.
  - Summary: `outputs/attacks/nuscenes_keyframe/nuscenes_voxelnext_no_defense_eval_summary.json`.
- E2.1 Injection VoxelNeXt no-defense is done:
  - Raw metric: `third_party/OpenPCDet/output/nuscenes_models/cbgs_voxel0075_voxelnext/E4_3_voxelnext_keyframe_no_defense_E2_1/eval/epoch_1/val/v1.0-trainval/final_result/data/metrics_summary.json`.
  - Parsed summary: `outputs/attacks/nuscenes_keyframe/nuscenes_voxelnext_no_defense_eval_summary.json`.
  - Result: mAP 21.97, NDS 41.87, mAP ASR 63.70% vs clean 60.52/66.64.
  - Attack-strength gate passed for Injection.
- E2.2 Perturbation VoxelNeXt no-defense is done:
  - Raw metric: `third_party/OpenPCDet/output/nuscenes_models/cbgs_voxel0075_voxelnext/E4_3_voxelnext_keyframe_no_defense_E2_2/eval/epoch_1/val/v1.0-trainval/final_result/data/metrics_summary.json`.
  - Parsed summary: `outputs/attacks/nuscenes_keyframe/nuscenes_voxelnext_no_defense_eval_summary.json`.
  - Result: mAP 51.87, NDS 59.95, mAP ASR 14.29% vs clean 60.52/66.64.
  - Interpretation: usable as a mild-to-moderate perturbation setting, but should be budget-gated at `epsilon=0.5` before the paper treats it as a stronger nuScenes attack.
- E2.3 Dropping VoxelNeXt no-defense is done:
  - Raw metric: `third_party/OpenPCDet/output/nuscenes_models/cbgs_voxel0075_voxelnext/E4_3_voxelnext_keyframe_no_defense_E2_3/eval/epoch_1/val/v1.0-trainval/final_result/data/metrics_summary.json`.
  - Parsed summary: `outputs/attacks/nuscenes_keyframe/nuscenes_voxelnext_no_defense_eval_summary.json`.
  - Result: mAP 58.91, NDS 65.76, mAP ASR 2.66% vs clean 60.52/66.64.
  - Interpretation: too weak for a main nuScenes Dropping robustness claim under the current keyframe-only `drop_ratio=0.5` setup.
- Decision gate update: pause TransFusion-Lidar and nuScenes defense matrices. First run a VoxelNeXt-only budget gate with stronger Perturbation (`epsilon=0.5`) and upper-bound Dropping (`drop_ratio=1.0`). If Dropping remains weak even at 1.0, treat it as a weak diagnostic or replace it with a sweep-aware/saliency-aware protocol.
- Started nuScenes budget-gate run in tmux:
  - Session: `asap_nusc_budget_gate_20260530_180031`.
  - Command: `NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_nuscenes_attack_budget_gate.sh`.
  - Log: `outputs/attacks/nuscenes_keyframe_budget_gate/asap_nusc_budget_gate_20260530_180031.log`.
  - Candidate outputs: `outputs/attacks/nuscenes_keyframe_budget_gate/E2.2_perturbation_eps0p5/` and `outputs/attacks/nuscenes_keyframe_budget_gate/E2.3_dropping_drop1p0/`.
  - Summary: `outputs/attacks/nuscenes_keyframe_budget_gate/nuscenes_voxelnext_budget_gate_eval_summary.json`.
- nuScenes budget-gate results:
  - Perturbation `epsilon=0.5`: VoxelNeXt mAP/NDS 40.27/51.46, mAP ASR 33.46%, NDS ASR 22.78%. Accepted as the nuScenes Perturbation transfer budget.
  - Dropping `drop_ratio=1.0`: VoxelNeXt mAP/NDS 55.58/63.91, mAP ASR 8.17%, NDS ASR 4.10%. Still weak; keep nuScenes Dropping paused for main claims.
  - Clean `LIDAR_TOP` restored after eval; both T4 GPUs idle after completion.
  - Decision: run TransFusion-Lidar no-defense next on accepted attacks only: Injection original and Perturbation `epsilon=0.5`.
- Started TransFusion-Lidar accepted no-defense gate in tmux:
  - Session: `asap_nusc_transfusion_accepted_20260530_200058`.
  - Command: `NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 DETECTOR=transfusion_lidar bash scripts/eval_nuscenes_accepted_no_defense.sh`.
  - Log: `outputs/attacks/nuscenes_keyframe_budget_gate/asap_nusc_transfusion_accepted_20260530_200058.log`.
  - Summary: `outputs/attacks/nuscenes_keyframe_budget_gate/nuscenes_transfusion_lidar_accepted_no_defense_eval_summary.json`.
  - E2.1 Injection result: mAP 19.13, NDS 39.45, mAP ASR 70.37% vs clean 64.57/69.43.
  - E2.2 Perturbation `epsilon=0.5` result: mAP 43.02, NDS 52.99, mAP ASR 33.37% vs clean 64.57/69.43.
  - Clean `LIDAR_TOP` restored after eval; both T4 GPUs idle after completion.
  - Interpretation: accepted nuScenes attacks transfer to both VoxelNeXt and TransFusion-Lidar. Dropping remains excluded from nuScenes main claims.
- Accepted SOR/ROR defense matrix completed in tmux session `asap_nusc_sor_ror_20260530_210112`.
  - Command: `FORCE_PURIFY=1 NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 DETECTOR=all DEFENSES=sor,ror bash scripts/eval_nuscenes_accepted_defenses.sh`.
  - Log: `outputs/defenses/nuscenes_keyframe_accepted/asap_nusc_sor_ror_20260530_210112.log`.
  - VoxelNeXt summary: `outputs/defenses/nuscenes_keyframe_accepted/nuscenes_voxelnext_accepted_defenses_eval_summary.json`.
    - Injection SOR: 30.01/46.64, mAP ASR 50.42%; Injection ROR: 53.29/62.66, mAP ASR 11.94%.
    - Perturbation `epsilon=0.5` SOR: 42.16/53.16, mAP ASR 30.34%; ROR: 51.36/60.40, mAP ASR 15.14%.
  - TransFusion-Lidar summary: `outputs/defenses/nuscenes_keyframe_accepted/nuscenes_transfusion_lidar_accepted_defenses_eval_summary.json`.
    - Injection SOR: 30.92/46.78, mAP ASR 52.12%; Injection ROR: 57.29/65.04, mAP ASR 11.28%.
    - Perturbation `epsilon=0.5` SOR: 43.95/53.94, mAP ASR 31.94%; ROR: 53.51/61.23, mAP ASR 17.14%.
  - Interpretation: ROR is a very strong non-ASAP baseline on all accepted nuScenes rows. ASAP must beat it or be positioned honestly as complementary/competitive; do not claim easy dominance.
  - Clean `data/nuscenes/v1.0-trainval/samples/LIDAR_TOP` restored after eval; it is a real directory, not a symlink.
- nuScenes Track A ASAP assets prepared.
  - Checkpoint: `checkpoints/nuscenes/asap_score_net_trackA_v1.pth`.
    - Built on 128 clean nuScenes keyframes, 8192 local patches, 10 epochs, `--num_features 5`, `--data_parallel` over both visible T4 GPUs.
    - Checkpoint loads on CPU and has no `module.` state-dict prefix.
  - Scorer configs: `outputs/asap_nuscenes_trackA_assets/configs/scorer_E2_1.json` and `outputs/asap_nuscenes_trackA_assets/configs/scorer_E2_2_eps0p5.json`.
    - E2.1 Injection calibration: 20 paired frames, 70678 clean SPUs, 1104 attacked SPUs, tau 0.014, `attack_kind=injection`.
    - E2.2 Perturbation eps0.5 calibration: 20 paired frames, 70678 clean SPUs, 4103 attacked SPUs, tau 0.049, `attack_kind=perturbation`.
  - Added scripts:
    - `scripts/prepare_nuscenes_tracka_assets.sh` for checkpoint/scorer preparation.
    - `scripts/smoke_nuscenes_tracka_asap.sh` for 2-frame dual-GPU smoke.
  - Smoke passed at `outputs/smoke/nuscenes_tracka_asap/`.
    - Injection smoke: 2 frames, 5D output valid, Injection filter enabled, ~10% point count reduction in the two sample frames.
    - Perturbation smoke: 2 frames, 5D output valid, Injection filter disabled, point count unchanged while score-net edited point coordinates.
- Started first full nuScenes ASAP run on VoxelNeXt only, to avoid blindly running the full detector matrix before seeing results.
  - tmux session: `asap_nusc_asap_voxelnext_20260531_005023`.
  - Command: `FORCE_PURIFY=1 NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 DETECTOR=voxelnext DEFENSES=asap ASAP_SCORER_JSON_DIR=outputs/asap_nuscenes_trackA_assets/configs ASAP_SCORE_NET_CKPT=checkpoints/nuscenes/asap_score_net_trackA_v1.pth bash scripts/eval_nuscenes_accepted_defenses.sh`.
  - Status at 2026-05-31 00:51 CST: running Injection purification with two `asap.pipeline` shards on GPUs 0 and 1. After purification, the script will run VoxelNeXt eval for Injection and Perturbation and parse `outputs/defenses/nuscenes_keyframe_accepted/nuscenes_voxelnext_accepted_defenses_eval_summary.json`.
  - Note: outer tee log was accidentally named `outputs/defenses/nuscenes_keyframe_accepted/.log`; per-GPU logs are correct under `outputs/defenses/nuscenes_keyframe_accepted/logs/asap_E2.1_injection_gpu*.log`.
  - Progress check commands:
    - `tmux capture-pane -pt asap_nusc_asap_voxelnext_20260531_005023 -S -120`
    - `find outputs/defenses/nuscenes_keyframe_accepted/asap/E2.1_injection/shards/output_gpu0 outputs/defenses/nuscenes_keyframe_accepted/asap/E2.1_injection/shards/output_gpu1 -maxdepth 1 -name '*.bin' | wc -l`
    - `find third_party/OpenPCDet/output/nuscenes_models -path '*accepted_asap*metrics_summary.json' -print`
- VoxelNeXt ASAP-v1 completed and parsed.
  - Combined summary restored at `outputs/defenses/nuscenes_keyframe_accepted/nuscenes_voxelnext_accepted_defenses_eval_summary.json` after adding `--merge-existing` to `scripts/parse_nuscenes_eval_summary.py`.
  - Injection ASAP-v1: 52.45/62.18, mAP ASR 13.33%. This is close to ROR 53.29/62.66, but still lower by 0.84 mAP and 0.48 NDS.
  - Perturbation `epsilon=0.5` ASAP-v1: 41.32/52.74, mAP ASR 31.73%. This is only slightly above no-defense 40.27/51.46 and far below ROR 51.36/60.40.
  - Meta summary:
    - Injection: mean input/output 36034/29635 points, average point reduction 17.76%, radius filter enabled, 38,520,748 total post-filter drops.
    - Perturbation: mean input/output unchanged at 34720 points, radius filter disabled, score-net edits about 1544 points/frame but does not recover enough detection accuracy.
  - Decision: do not launch TransFusion-Lidar ASAP blindly. First run a perturbation-focused VoxelNeXt diagnostic.
- Started perturbation-focused diagnostic: ASAP score-net plus radius filter enabled for `perturbation`.
  - Code support added:
    - `PurifierConfig.score_filter_attack_kinds` and CLI `--score_filter_attack_kinds`, default `injection` to preserve historical behavior.
    - `scripts/eval_nuscenes_accepted_defenses.sh` now supports `ATTACKS=<list>` and `TAG_SUFFIX=<suffix>`, and uses `--merge-existing` when parsing.
    - `scripts/smoke_nuscenes_tracka_asap.sh` now supports `DRY_RUN` and passes `--score_filter_attack_kinds`.
  - Smoke: `outputs/smoke/nuscenes_tracka_asap_pfilter/` passed for 1 frame; Perturbation pfilter dropped 3565 / 34720 points in that frame and kept valid 5D output.
  - tmux session: `asap_nusc_pfilter_voxelnext_20260531_162416`.
  - Command intent: only `ATTACKS=E2_2_eps0p5`, `DETECTOR=voxelnext`, `DEFENSES=asap`, `TAG_SUFFIX=_pfilter`, `DEFENSE_ROOT=outputs/defenses/nuscenes_keyframe_accepted_asap_pfilter`, `ASAP_SCORE_FILTER_ATTACK_KINDS=injection,perturbation`.
  - Status at launch check: running with two `asap.pipeline` shards; output count about 145 / 6019 frames.
- Perturbation pfilter diagnostic completed and validated.
  - Parsed summary: `outputs/defenses/nuscenes_keyframe_accepted_asap_pfilter/nuscenes_voxelnext_pfilter_eval_summary.json`.
  - Successful OpenPCDet log: `third_party/OpenPCDet/output/nuscenes_models/cbgs_voxel0075_voxelnext/E4_3_voxelnext_accepted_asap_E2_2_eps0p5_pfilter/eval/epoch_1/val/v1.0-trainval/log_eval_20260531-183502.txt`; it ends with `Evaluation done`.
  - Earlier outer log `outputs/defenses/nuscenes_keyframe_accepted_asap_pfilter/asap_nusc_pfilter_voxelnext_20260531_162416.log` contains a failed eval with missing `LIDAR_TOP` files, but the later retry under the same OpenPCDet tag succeeded and wrote metrics at 2026-05-31 18:50.
  - Result: VoxelNeXt Perturbation `epsilon=0.5` ASAP-pfilter mAP/NDS **52.27/61.23**, mAP ASR **13.64%**, NDS ASR **8.13%**.
  - Comparison: no-defense 40.27/51.46; ASAP-v1 41.32/52.74; ROR 51.36/60.40. Pfilter improves over ASAP-v1 by +10.95 mAP/+8.49 NDS and exceeds ROR by +0.91 mAP/+0.83 NDS.
  - Meta: 6019 metadata rows; output has 6019 `.bin` files and no broken symlinks. `score_filter_attack_kinds=['injection', 'perturbation']`; all frames have radius-filter drops; total dropped points 34,759,558, mean 5775/frame, 16.63% of input points.
  - Interpretation: this is a score-net + ROR-like radius-filter cascade. It is a promising revised perturbation policy, but not evidence that pure learned diffusion alone dominates ROR.
- TransFusion-Lidar perturbation pfilter transfer check completed.
  - tmux session: `asap_nusc_pfilter_transfusion_20260531_191355`.
  - It reused the existing 6019-frame pfilter output and did not rerun purification.
  - Parsed summaries:
    - `outputs/defenses/nuscenes_keyframe_accepted_asap_pfilter/nuscenes_transfusion_lidar_accepted_defenses_eval_summary.json`
    - `outputs/defenses/nuscenes_keyframe_accepted_asap_pfilter/nuscenes_transfusion_lidar_pfilter_eval_summary.json`
  - Raw metrics: `third_party/OpenPCDet/output/nuscenes_models/transfusion_lidar/E4_4_transfusion_lidar_accepted_asap_E2_2_eps0p5_pfilter/eval/epoch_no_number/val/v1.0-trainval/final_result/data/metrics_summary.json`.
  - Result: TransFusion-Lidar Perturbation `epsilon=0.5` ASAP-pfilter mAP/NDS **54.40/61.98**, mAP ASR **15.75%**, NDS ASR **10.73%**.
  - Comparison: no-defense 43.02/52.99; ROR 53.51/61.23. Pfilter exceeds ROR by +0.89 mAP/+0.75 NDS.
  - Clean `data/nuscenes/v1.0-trainval/samples/LIDAR_TOP` restored after completion; it is a real directory, not a symlink.
- TransFusion-Lidar Injection ASAP-v1 result is available and parsed.
  - Raw metrics: `third_party/OpenPCDet/output/nuscenes_models/transfusion_lidar/E4_4_transfusion_lidar_accepted_asap_E2_1/eval/epoch_no_number/val/v1.0-trainval/final_result/data/metrics_summary.json`.
  - Parsed into `outputs/defenses/nuscenes_keyframe_accepted/nuscenes_transfusion_lidar_accepted_defenses_eval_summary.json` with `--merge-existing`.
  - Result: TransFusion-Lidar Injection ASAP-v1 mAP/NDS **56.74/64.72**, mAP ASR **12.12%**, NDS ASR **6.79%**.
  - Comparison: ROR is slightly higher at 57.29/65.04, so phrase this as near-parity with ROR, not dominance.
- Paper-facing table strategy updated:
  - Use `ASAP-family result` as the table column.
  - Tag perturbation rows using the cascade as `ASAP-pfilter`.
  - Do not leave TransFusion rows as pending.

## Recommended next steps
1. Assemble the section Markdown and LaTeX fragments into a first IEEE-style ICASSP manuscript draft.
2. Run a reviewer-style pass over the assembled draft, focused on pfilter/ROR novelty, keyframe-only nuScenes caveats, runtime, and whether the claims stay below the evidence.
3. Keep new GPU experiments paused unless the review pass identifies a specific acceptance-critical gap. Do not spend GPU time on nuScenes Dropping defense rows unless a stronger sweep-aware/saliency-aware dropping protocol is implemented.
4. Treat revised Track A as the current KITTI two-detector candidate, but phrase claims as cross-attack robustness rather than per-row superiority.
5. Use `tmux` for long jobs and both T4 GPUs where safe. `eval_attacked_kitti.sh` supports `NUM_GPUS=2`; `run_asap_vpsde_score_net_dual_gpu.sh` shards score-net purification across GPUs.
6. Investigate sparsity-aware thresholds or class-aware calibration only after the first assembled paper draft passes reviewer-style consistency review.

## Manuscript cleanup completed on 2026-05-31
- Submission-facing sections `docs/paper/00_abstract.md` through `05_conclusion.md` now have no HTML comments, TODO markers, placeholders, or missing citation keys.
- `docs/paper/references.bib` was rewritten with verified entries for datasets, detectors, attacks, defenses, diffusion purification, LiDAR-SPD, PointGuard, and ASAP math references.
- Added `docs/paper/fig_asap_pipeline.tex` and `docs/paper/tables_icassp.tex` as LaTeX-ready figure/table assets.
- Added `scripts/check_paper_integrity.py`; `python scripts/check_paper_integrity.py` currently passes.

## E6 cost/selectivity update completed on 2026-05-31
- Added `scripts/analyze_cost_metrics.py`.
- Generated:
  - `outputs/cost/E6_summary/cost_selective_summary.csv`
  - `outputs/cost/E6_summary/cost_selective_summary.json`
  - `outputs/cost/E6_summary/cost_selective_summary.md`
- Key nuScenes Perturbation pfilter numbers:
  - 6019 frames.
  - M2 flagged SPU ratio: 22.75%.
  - Score-net SPU ratio: 18.87%.
  - Edited point ratio before support filter: 4.78%.
  - Support-filter / output drop ratio: 16.63%.
  - Log-derived two-T4 aggregate preprocessing throughput: about 0.48 s/frame.
- Paper §4.5 now uses these numbers to explain selectivity and to disclose that pfilter gains come from a score-net + support-filter cascade, not pure score-net purification alone.

## Innovation/SOTA gap response started on 2026-05-31
- User correctly flagged that method novelty and experiment completeness are still insufficient for a strong ICASSP submission.
- Added `docs/10_innovation_and_sota_upgrade_plan.md`.
- Added M1 CLI hyperparameter controls to `src/asap/pipeline/__init__.py`:
  - `--spu_k`, `--spu_alpha`, `--spu_beta`, `--spu_r_min`, `--spu_r_max`, `--spu_eta`, `--spu_max_centers`, `--spu_min_points_per_spu`.
- Added SPD-style baseline scripts:
  - `scripts/run_spd_style_uniform_score_net_dual_gpu.sh`
  - `scripts/eval_spd_style_uniform_kitti.sh`
- SPD-style baseline definition:
  - fixed radius `r1=0.40 m`, `r2=0.67*r1`, `eta=0.40 m`;
  - `tau=-1.0`, so every SPU is purified;
  - same Track A score-net checkpoint and `score_anchor_blend=0.75`.
- 20-frame smoke on KITTI E2.2 Perturbation:
  - 100% SPUs flagged;
  - 97.45% SPUs enter score-net;
  - 77.81% points edited;
  - single-T4 runtime about 4.74 s/frame.
- Full KITTI E2.2 Perturbation SPD-style baseline was started in tmux session `asap_spd_uniform_e22_20260531_2316` and later completed; see the parsed result below.

## SPD-style proxy result parsed on 2026-06-01
- Full KITTI E2.2 Perturbation fixed-SPU all-unit proxy completed:
  - 3769/3769 purified frames.
  - PointPillars result: **57.16 / 30.35** Car/Moderate AP_R40 / cross-class mAP.
  - Baselines for interpretation:
    - No defense: 56.54 / 27.32.
    - ROR: 41.65 / 19.46.
    - ASAP Track A: 61.04 / 34.69.
  - Full-run metadata:
    - 100.00% SPUs flagged by design.
    - 97.14% SPUs enter score-net.
    - 77.86% points edited.
    - Effective two-T4 preprocessing throughput about 2.39 s/frame; per-shard logs about 4.78 s/frame.
- Decision: P0 supports ASAP's novelty argument. Fixed-radius nonselective diffusion is much heavier and weaker than adaptive SPU + M2 selective purification. Keep it as an innovation ablation and do not reframe the method around fixed-unit diffusion.
- P2 adaptive-SPU all-unit result:
  - Full KITTI E2.2 Perturbation adaptive-radius all-SPU proxy completed on 2026-06-01.
  - PointPillars result: **38.62 / 17.88** Car/Moderate AP_R40 / cross-class mAP.
  - Metadata: 100.00% SPUs flagged, 95.92% SPUs enter score-net, 70.98% points edited.
  - Effective two-T4 preprocessing throughput about 3.62 s/frame; per-shard logs about 7.18--7.31 s/frame.
  - Interpretation: M2 is a core structure-preservation gate, not just compute allocation. Removing M2 causes severe over-purification and is worse than no defense.
- P1 support-only vs pfilter interpretation:
  - nuScenes ROR uses the same `radius=0.4, min_neighbors=5` rule as pfilter but applies it directly to attacked scans, so it is the support-only baseline.
  - VoxelNeXt Perturbation: support-only/ROR 51.36/60.40; score-net only 41.32/52.74; score-net+support 52.27/61.23.
  - TransFusion-Lidar Perturbation: support-only/ROR 53.51/61.23; score-net+support 54.40/61.98.
  - Claim boundary: pfilter gain is a small but repeatable cascade gain over support-only, not pure score-net dominance.
- Tau sensitivity completed:
  - Script: `scripts/run_kitti_e22_tau_sensitivity.sh`.
  - Current Track A `tau=0.024`: **61.04 / 34.69**.
  - `tau=0.036`: **59.88 / 30.21**.
  - `tau=0.048`: **58.02 / 28.79**.
  - Decision: keep `tau=0.024`. Stricter selection lowers cross-class mAP, so the paper should not tune thresholds further for this submission. Use P2 as evidence that M2 is a structure-preservation gate, not simply a knob for editing fewer SPUs.

## Paper mainline update on 2026-06-01
- Updated `docs/paper/01_introduction.md` to describe C3 as anomaly-selective structure preservation.
- Updated `docs/paper/03_method.md` to foreground M2 as an anomaly-selective structure gate and to define ASAP-pfilter as a score-net + support-filter cascade.
- Updated `docs/paper/04_experiments.md` with the tau sensitivity result and final threshold decision.
- Updated `docs/09_icassp_acceptance_gap_audit.md` and `docs/10_innovation_and_sota_upgrade_plan.md` so P0/P1/P2/P4 are treated as completed decision evidence.
- `python scripts/check_paper_integrity.py` passes after these edits.
- Next mainline task: assemble the first IEEE-style manuscript package and run a reviewer-style audit focused on novelty versus ROR/SPD-style proxies, keyframe-only nuScenes caveats, runtime, and claim control.

## GPU utilization / runtime optimization note on 2026-06-01
- User observed low GPU memory usage (~150--300 MB) and high CPU/RAM usage. Diagnosis:
  - During OpenPCDet evaluation, low GPU memory often occurs in dataloader / CPU post-processing / KITTI AP calculation phases.
  - ASAP purification itself is CPU-heavy because M1 SPU construction uses SciPy `cKDTree`, M2 feature extraction uses NumPy/Python loops/eigendecomposition, support filtering uses `cKDTree`, and score-net inference was previously launched one SPU at a time.
- Implemented first low-risk GPU utilization improvement:
  - Added `PurifierConfig.score_batch_spus` and CLI `--score_batch_spus` with default `64`.
  - Batched score-net forward passes for SPUs with identical inner point counts.
  - Updated run scripts to pass `SCORE_BATCH_SPUS`.
  - Smoke comparison on 8 KITTI E2.2 frames:
    - batch=1 wall time: 25.293 s.
    - batch=64 wall time: 13.227 s.
    - output max absolute difference: `3.8147e-06`, mean absolute difference about `8.7e-10`.
  - This is a safe speedup but does not solve the biggest bottleneck: KD-tree/SPU/M2 remain CPU-bound.
- Installed GPU geometry libraries are currently missing: `pytorch3d`, `torch_cluster`, `torch_geometric`, `cupy`, `faiss`.
- Recommended next optimization slice:
  1. Add profiling logs around M1, M2, M3, support filter, I/O.
  2. Implement an optional torch-GPU SPU path for KITTI-sized clouds using `torch.cdist` chunking and batched top-k/radius masks.
  3. Only after accuracy equivalence is verified, use it for long experiments.
- Profiling / GPU M1 follow-up:
  - Added `--profile` to `python -m asap.pipeline`; meta JSONL now records `profile_io_read_ms`, `profile_m1_ms`, `profile_m2_ms`, `profile_m3_ms`, `profile_io_write_ms`, and totals.
  - 20-frame KITTI E2.2 baseline after score-net batching:
    - M1 mean 491.9 ms/frame.
    - M2 mean 592.5 ms/frame.
    - M3 mean 200.4 ms/frame.
    - Total mean 1298.0 ms/frame.
  - Added experimental `--spu_backend torch` with torch cdist chunking.
  - Torch M1 prototype result on same 20 frames:
    - M1 mean 2764.9 ms/frame, total mean 3568.5 ms/frame.
    - Slower than SciPy cKDTree and has small boundary-induced output drift (`max_abs` 0.51 on coordinates, mean abs `4.3e-05`).
  - Decision: do not use naive torch-cdist M1 for formal experiments. Keep it as an experimental backend only. Next practical optimization should target M2 vectorization or a specialized point-cloud radius-query library, not dense cdist.
- M2 vectorization completed:
  - `compute_features` now groups SPUs by identical `len(inner_idx)` and batch-computes PCA anisotropy and vMF concentration with dense NumPy arrays.
  - Numerical equivalence:
    - Single-frame feature max abs diff vs old implementation: `8.79e-14`.
    - 20-frame purified output diff vs old implementation: max abs `0`.
  - 20-frame KITTI E2.2 profile after M2 batching:
    - M1 mean 492.1 ms/frame.
    - M2 mean 63.5 ms/frame, down from 592.5 ms/frame.
    - M3 mean 189.2 ms/frame.
    - Total mean 754.6 ms/frame, down from 1298.0 ms/frame.
  - This is the current best safe optimization. It preserves results exactly on the smoke set and should be used for future runs.
