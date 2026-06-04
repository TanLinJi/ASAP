# 4. Experiments

## 4.1 Experimental setup

We evaluate ASAP as an inference-only front-end: every detector is used with its released checkpoint and no weight modification. KITTI [@geiger2012kitti] is used for controlled ablations with PointPillars [@lang2019pointpillars] and PV-RCNN [@shi2020pvrcnn]. nuScenes [@caesar2020nuscenes] tests transfer to modern 360-degree detectors, VoxelNeXt [@chen2023voxelnext] and TransFusion-Lidar [@bai2022transfusion]. KITTI covers point injection, perturbation, and dropping; nuScenes uses accepted keyframe-only Injection and `epsilon=0.5` Perturbation attacks because OpenPCDet evaluates ten sweeps and clean historical sweeps weaken keyframe-only Dropping. KITTI reports `Car Moderate AP_R40 / Moderate mAP` over Car/Pedestrian/Cyclist; nuScenes reports `mAP / NDS`.

## 4.2 Baselines and policy variants

Baselines include no defense and global SOR/ROR [@rusu2008towards]. We report **ASAP-v1** for the score-net purifier with M1/M2 gating, and **ASAP-pfilter** when the disclosed support filter is appended after score-net denoising. For recent-method context, we add a same-protocol **LiDAR-SPD-style proxy** [@lidarspd] using the reported fixed radius `r_1=0.15 m`; this is a controlled proxy, not an official reproduction. PWAVEP [@li2026pwavep] and the CCS 2025 real-time LiDAR defense [@yan2025realtime] are discussed as recent related methods, but are not placed in the result table because their released/available paths do not provide directly comparable full-scene KITTI detection outputs under our attack protocol. The pfilter row is a cascade, not a claim that diffusion alone dominates ROR.

## 4.3 Main results

Table 1 reports the clean upper bound, no-defense lower bound, strongest non-ASAP baseline, and ASAP-family result. Higher values are better. The clean row is only a reference; ASAP is not expected to exceed it.

TABLE 1 (current verified rows):

| Dataset  | Detector            | Detector family                                | Attack       | Clean ref | No defense | Best non-ASAP baseline | **ASAP-family result** |
|----------|---------------------|------------------------------------------------|--------------|----------:|-----------:|------------------------:|----------------:|
| KITTI    | PointPillars        | Pillar-based one-stage                         | Injection    | 78.40 / 64.21 | 60.10 / 38.73 | 66.07 / 51.30 (ROR) | 65.71 / **52.18** |
| KITTI    | PointPillars        | Pillar-based one-stage                         | Perturbation | 78.40 / 64.21 | 56.54 / 27.32 | 41.65 / 19.46 (ROR) | **61.26 / 34.76** |
| KITTI    | PointPillars        | Pillar-based one-stage                         | Dropping     | 78.40 / 64.21 | 71.22 / 48.42 | 49.91 / 34.20 (ROR) | **71.22 / 48.42** |
| KITTI    | PV-RCNN             | Voxel-point two-stage                          | Injection    | 84.36 / 69.74 | 63.16 / 48.22 | **72.77 / 57.77** (ROR) | 72.59 / 57.73 |
| KITTI    | PV-RCNN             | Voxel-point two-stage                          | Perturbation | 84.36 / 69.74 | 3.50 / 2.45 | 1.94 / 1.17 (ROR) | **11.23 / 6.26** |
| KITTI    | PV-RCNN             | Voxel-point two-stage                          | Dropping     | 84.36 / 69.74 | 78.61 / 57.12 | 59.05 / 43.12 (ROR) | **78.61 / 57.12** |
| nuScenes | VoxelNeXt (0.075)   | Fully sparse voxel CNN, anchor-free            | Injection    | 60.52 / 66.64 | 21.97 / 41.87 | 53.29 / 62.66 (ROR) | 52.45 / 62.18 |
| nuScenes | VoxelNeXt (0.075)   | Fully sparse voxel CNN, anchor-free            | Perturbation (`eps=0.5`) | 60.52 / 66.64 | 40.27 / 51.46 | 51.36 / 60.40 (ROR) | **52.27 / 61.23** (`ASAP-pfilter`) |
| nuScenes | TransFusion-Lidar   | Sparse voxel backbone + transformer query head | Injection    | 64.57 / 69.43 | 19.13 / 39.45 | **57.29 / 65.04** (ROR) | 56.74 / 64.72 |
| nuScenes | TransFusion-Lidar   | Sparse voxel backbone + transformer query head | Perturbation (`eps=0.5`) | 64.57 / 69.43 | 43.02 / 52.99 | 53.51 / 61.23 (ROR) | **54.40 / 61.98** (`ASAP-pfilter`) |

ASAP is strongest on perturbation. On KITTI PointPillars it improves the attacked row from **56.54 / 27.32** to **61.26 / 34.76**, and on KITTI PV-RCNN it improves the same attack from **3.50 / 2.45** to **11.23 / 6.26**. On nuScenes Perturbation, ASAP-pfilter improves VoxelNeXt from **40.27 / 51.46** to **52.27 / 61.23** and TransFusion-Lidar from **43.02 / 52.99** to **54.40 / 61.98**, slightly above the same global support rule used by ROR on both detectors. Injection is near ROR rather than dominant, and Dropping is handled conservatively: when evidence has already been removed, ASAP skips M3 rather than claiming reconstruction. The nuScenes Dropping attack is excluded because it fails the attack-strength gate under ten-sweep evaluation.

## 4.4 Policy ablations

Table 2 keeps only ablations that affect the paper's main claims: whether selective local purification matters, whether the support branch is merely ROR, and whether the attack-aware Dropping policy avoids over-editing.

TABLE 2 (verified policy ablations):

| Setting | Detector / attack | Result | Interpretation |
|---------|-------------------|--------|----------------|
| No defense | PointPillars / Perturbation | 56.54 / 27.32 | Attacked lower bound. |
| LiDAR-SPD-style proxy | PointPillars / Perturbation | 56.58 / 26.61 | Fixed `r=0.15 m`, `tau=-1.0`; edits 64.40% of points and remains near no defense. |
| Fixed-SPU all-unit diffusion proxy | PointPillars / Perturbation | 57.16 / 30.35 | Fixed `r=0.40 m`, `tau=-1.0`; edits 77.86% of points and remains below ASAP. |
| Adaptive-SPU all-unit diffusion | PointPillars / Perturbation | 38.62 / 17.88 | ASAP M1 with `tau=-1.0`; edits 70.98% of points and collapses without M2 selection. |
| **ASAP tstar008** | PointPillars / Perturbation | **61.26 / 34.76** | Adaptive SPUs plus M2 selection outperform nonselective diffusion; `t_star=0.08` gives a small positive update over Track A. |
| LiDAR-SPD-style proxy | PV-RCNN / Perturbation | 3.16 / 2.31 | Same purified scans transferred to a two-stage detector; below no defense and ASAP Track A. |
| **ASAP tstar008** | PV-RCNN / Perturbation | **11.23 / 6.26** | Selective editing recovers signal under the stronger PV-RCNN perturbation attack, though absolute recovery remains limited. |
| No defense | VoxelNeXt / Perturbation | 40.27 / 51.46 | Strong accepted nuScenes perturbation budget. |
| ASAP-v1 | VoxelNeXt / Perturbation | 41.32 / 52.74 | Score-net editing alone is insufficient under ten-sweep nuScenes evaluation. |
| ROR / support-only | VoxelNeXt / Perturbation | 51.36 / 60.40 | Same `0.4 m / 5 neighbors` support rule as pfilter, applied to the attacked scan. |
| **ASAP-pfilter** | VoxelNeXt / Perturbation | **52.27 / 61.23** | Score-net + support filter exceeds support-only by +0.91 mAP / +0.83 NDS. |
| ROR / support-only | TransFusion-Lidar / Perturbation | 53.51 / 61.23 | Same support-only row transfers across detector heads. |
| **ASAP-pfilter** | TransFusion-Lidar / Perturbation | **54.40 / 61.98** | The cascade exceeds support-only by +0.89 mAP / +0.75 NDS. |
| Score-net policy | PointPillars / Dropping | 68.49 / 35.41 | Extra editing removes useful evidence. |
| **Skip M3** | PointPillars / Dropping | **71.22 / 48.42** | Passing through the damaged scan is the safer conservative policy. |

These rows support M2 as a structure-preservation gate: nonselective diffusion either remains weak or collapses, even with the same score-net checkpoint. The LiDAR-SPD-style proxy is especially important because it uses the closest fixed spherical radius reported by a recent ICASSP defense, yet obtains **56.58 / 26.61** versus ASAP's **61.26 / 34.76** on PointPillars while editing **64.40%** of points. On PV-RCNN, the same proxy obtains only **3.16 / 2.31**, below no defense **3.50 / 2.45** and far below ASAP **11.23 / 6.26**. The pfilter rows show the support filter is the dominant nuScenes perturbation component, but score-net preconditioning adds a small detector-transferable gain over the same support rule. Two corrective ablations bound the claim. First, fixed `r=0.40` slightly exceeds the original adaptive Track A on KITTI Perturbation (**61.06 / 34.92** vs **61.04 / 34.69**) but remains close to the current `t_star=0.08` row and edits more points (**27.07%** vs **21.93%**), so M1 is a conservative locality prior, not strict AP dominance. Second, `f_d` alone nearly matches full M2 on KITTI Perturbation, so the four-feature scorer is claimed for cross-attack coverage rather than feature necessity on every attack.

## 4.5 Inference cost

ASAP is currently a robustness prototype, not a real-time system. On nuScenes Perturbation, SOR and ROR process the validation split in 192 s and 348 s, respectively; unoptimized ASAP-pfilter runs at about **0.48 s/frame** aggregate throughput over two T4 GPUs. The M2 gate remains selective: **22.75%** of candidate SPUs are flagged, **18.87%** enter the score-net path, and **4.78%** of points are coordinate-edited before the support filter removes **16.63%**. This cost is higher than ROR, but it is detector-agnostic, requires no retraining, and is much less destructive than all-unit diffusion proxies.
