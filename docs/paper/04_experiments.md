# 4. Experiments

<!-- 目标长度: 1 - 1.5 栏，1 张主表 + 1 张消融表。 -->

## 4.1 Experimental setup

<!--
- 数据集分层:
  1) KITTI: 快速复现、消融、与旧防御方法公平对比。
  2) Waymo / nuScenes: 支撑 detector-agnostic + recent SOTA claim。
- 检测器分层:
  1) KITTI sanity check: PointPillars, PV-RCNN++。
  2) Waymo modern detectors: DSVT-Voxel, VoxelNeXt, MPPNet。
  3) nuScenes modern detectors: VoxelNeXt, TransFusion-Lidar。
- 评测指标:
  KITTI: 3D Average Precision (AP) on Car/Pedestrian/Cyclist.
  Waymo: mean Average Precision / mean Average Precision weighted by Heading (mAP / mAPH).
  nuScenes: mean Average Precision (mAP) and nuScenes Detection Score (NDS).
- 攻击设置：至少 1 个 injection、1 个 perturbation、1 个 dropping 代表，强度统一。
-->

We evaluate ASAP on a tiered set of datasets and detectors so that the same purification module is stressed both in a controlled ablation regime and against recent state-of-the-art detectors. Throughout, ASAP is **inference-only**: every detector is used as released, with its official pre-trained checkpoint and no parameter modification.

- **Datasets.** KITTI [@geiger2012kitti] is used for fast reproducibility and ablation, since it is small enough to iterate on the local CUDA 12.4 setup. The Waymo Open Dataset [@sun2020waymo] and nuScenes [@caesar2020nuscenes] are used for the main detector-agnostic evaluation, because most recent strong LiDAR detectors only report on these two benchmarks.
- **KITTI detectors.** PointPillars and PV-RCNN serve as controlled baselines that are well understood by the community and easy to compare against prior defenses.
- **Waymo detectors.** We pick DSVT-Voxel, VoxelNeXt, and MPPNet to span three modern design families: dynamic sparse voxel transformers, fully sparse voxel detectors, and temporal multi-frame detectors.
- **nuScenes detectors.** We use VoxelNeXt and TransFusion-Lidar to stress ASAP on a different benchmark, a 360-degree LiDAR layout, and a transformer/query-based detection head.
- **Attacks.** We cover the three canonical attack families on point-cloud detection: a *point-injection* attack [@cao2019adversarial; @tu2020physically], a *point-perturbation* attack [@xiang2019pointcloud], and a *point-dropping* attack [@zheng2019pointcloud], each at a unified budget per detector.
- **Metrics.** KITTI reports 3D Average Precision (AP) on Car/Pedestrian/Cyclist; Waymo reports mean Average Precision and mean Average Precision weighted by Heading (mAP / mAPH); nuScenes reports mAP and the nuScenes Detection Score (NDS). For attack effectiveness we additionally report the **Attack Success Rate (ASR)** before and after defense, where ASR = 1 - (mAP-under-attack / mAP-clean) for the corresponding detector and dataset.
- **Hardware.** KITTI development runs on a local 2-GPU CUDA 12.4 node; Waymo / nuScenes evaluation uses official pre-trained checkpoints from the OpenPCDet model zoo so that no detector is retrained for ASAP.

## 4.2 Baselines

<!--
- No defense（原始攻击下检测）。
- 经典统计去噪：SOR / ROR。
- Uniform SPU diffusion (ours, no gating)：作为公平消融，证明 selective gating 是关键。
- 可选：再加一种 published purification 作为外部对照（按字数允许补充）。
-->

We compare ASAP against four baselines that together isolate the contribution of each design decision in ASAP:

- **No defense.** The attacked LiDAR scan is fed directly to the detector. This lower-bound row quantifies the raw vulnerability of each detector under each attack.
- **SOR / ROR** [@rusu2008towards]. Statistical Outlier Removal and Radius Outlier Removal applied to the *whole* scan. These training-free, detector-agnostic denoisers are the closest classical analog of ASAP and let us isolate the value of the diffusion step.
- **Uniform SPU diffusion (ours, no M2 gating).** ASAP with the anomaly scorer disabled, i.e. the VP-SDE purifier is applied to *every* SPU regardless of the score. Comparing this baseline against full ASAP isolates the contribution of the anomaly-guided selection (M2 + threshold $\tau$).
- **Scene-wide diffusion purifier** [@sun2023ada3diff]. A representative published diffusion-based point-cloud purifier applied to the entire scan, included for external context where checkpoints are available.

## 4.3 Main results

<!--
- 表 1：主结果不再只放 KITTI/PointPillars，而是以 detector 为行。
- 建议列: Dataset, Detector, Attack, No defense, Uniform SPU, ASAP, Clean upper bound。
- KITTI 详细 Car/Pedestrian/Cyclist AP 可以放补充表或附表。
- Waymo / nuScenes 用 mAP/mAPH/NDS 支撑 modern detector-agnostic claim。
-->

Table 1 reports the main detector-agnostic results. Rows are indexed by *(dataset, detector, attack family)* and columns report the clean upper bound, the no-defense lower bound, the strongest non-ASAP baseline, and full ASAP. The headline metrics are dataset-native (AP for KITTI, mAP / mAPH for Waymo, mAP / NDS for nuScenes) so that each row is directly comparable to the published numbers of the corresponding detector.

*Reading the table.* Higher detection metrics are better; lower ASR is better. We highlight in bold the best per-row defense; **ASAP** is reported in the last column. The clean upper bound is the official released number under the same protocol, included only as a reference; ASAP is not expected to exceed it.

TABLE 1 (to be filled after MS2-MS3 of the experiment plan):

| Dataset | Detector | Detector family | Attack | Clean (ref) | No defense | Best non-ASAP baseline | **ASAP (ours)** |
|---------|----------|-----------------|--------|-------------|------------|------------------------|-----------------|
| KITTI    | PointPillars       | Pillar-based one-stage             | Injection    | [XX.X] AP       | [XX.X] AP       | [XX.X] AP       | **[XX.X] AP**       |
| KITTI    | PV-RCNN            | Voxel-point two-stage              | Injection    | [XX.X] AP       | [XX.X] AP       | [XX.X] AP       | **[XX.X] AP**       |
| Waymo    | DSVT-Voxel         | Dynamic sparse voxel transformer   | Injection    | [XX.X]/[XX.X]   | [XX.X]/[XX.X]   | [XX.X]/[XX.X]   | **[XX.X]/[XX.X]**   |
| Waymo    | VoxelNeXt          | Fully sparse voxel detector        | Injection    | [XX.X]/[XX.X]   | [XX.X]/[XX.X]   | [XX.X]/[XX.X]   | **[XX.X]/[XX.X]**   |
| Waymo    | MPPNet             | Temporal multi-frame detector      | Injection    | [XX.X]/[XX.X]   | [XX.X]/[XX.X]   | [XX.X]/[XX.X]   | **[XX.X]/[XX.X]**   |
| nuScenes | VoxelNeXt          | Fully sparse voxel detector        | Injection    | [XX.X]/[XX.X]   | [XX.X]/[XX.X]   | [XX.X]/[XX.X]   | **[XX.X]/[XX.X]**   |
| nuScenes | TransFusion-Lidar  | Transformer/query detection head   | Injection    | [XX.X]/[XX.X]   | [XX.X]/[XX.X]   | [XX.X]/[XX.X]   | **[XX.X]/[XX.X]**   |

Perturbation and dropping attacks follow the same row layout in the supplementary table; we expect a similar ranking trend across attack families since ASAP's anomaly score is not tied to any single attack signature.

## 4.4 Ablation study

<!--
- A1: 固定半径 vs M1 自适应半径。
- A2: 单一异常特征 vs M2 全部 4 个特征组合。
- A3: 全部扩散 vs M3 阈值控制选择性扩散。
- A4: 阈值 tau 的敏感性曲线。
-->

All ablations are run on KITTI with a single detector (PointPillars) to keep the design space tractable; each ablation isolates one module of ASAP while keeping the other two at their defaults from Section 3.

- **A1 — Adaptive vs fixed SPU radius (M1).** Replace the density-adaptive radius rule of Equation (M1.1) with a single global radius $r_1$ swept over $\{0.10, 0.20, 0.30, 0.40\}$ m, keeping M2 and M3 unchanged. The expected message is that no single fixed radius matches the detection accuracy of the adaptive rule across both nearby dense surfaces and far-away sparse returns.
- **A2 — Full anomaly scorer vs single feature (M2).** Disable three of the four features at a time and keep only $f_c$, $f_a$, $f_v$, or $f_d$. This shows whether any single feature is sufficient, and quantifies the marginal value of combining compactness, anisotropy, vMF concentration, and density ratio.
- **A3 — Selective vs uniform diffusion (M3).** Compare full ASAP against the *Uniform SPU diffusion* baseline introduced in Section 4.2. This is the cleanest test that M2-driven gating, and not the diffusion step alone, is what preserves benign geometry.
- **A4 — Sensitivity to the anomaly threshold $\tau$.** Sweep $\tau$ along its ROC curve from $\tau = 0$ (everything purified) to $\tau \to 1$ (nothing purified) and plot detection accuracy and ASR. We expect a wide plateau around $\tau^{\star}$ (the Youden-$J$ threshold of Section 3.4.6), which would empirically support that $\tau$ does not need per-detector tuning.

TABLE 2 (to be filled after MS3):

| Variant            | Module touched | KITTI 3D AP (Car) | ASR    |
|--------------------|----------------|-------------------|--------|
| Full ASAP          | -              | [XX.X]            | [XX.X] |
| A1 — fixed $r_1$   | M1             | [XX.X]            | [XX.X] |
| A2 — only $f_c$    | M2             | [XX.X]            | [XX.X] |
| A2 — only $f_a$    | M2             | [XX.X]            | [XX.X] |
| A2 — only $f_v$    | M2             | [XX.X]            | [XX.X] |
| A2 — only $f_d$    | M2             | [XX.X]            | [XX.X] |
| A3 — uniform SPU   | M3             | [XX.X]            | [XX.X] |
| A4 — $\tau = \tau^{\star}/2$ | M2 threshold | [XX.X]   | [XX.X] |
| A4 — $\tau = 2\tau^{\star}$  | M2 threshold | [XX.X]   | [XX.X] |

## 4.5 Inference cost

<!--
- 报告：clean 推理时间 / 受扰未净化时间 / 全扩散净化时间 / ASAP 选择性净化时间。
- 单位 ms per frame，单卡 GPU。
- 强调 selective gating 的加速倍率。
-->

We report wall-clock inference cost per frame on a single GPU, separated into (i) the detector's own forward pass on the *attacked* scan, (ii) the additional purification cost of each defense, and (iii) the resulting end-to-end latency. Costs are averaged over 500 frames of the KITTI validation split with the PointPillars detector.

TABLE 3 (to be filled after MS4):

| Pipeline                                   | Purification cost (ms) | Detector cost (ms) | End-to-end (ms) | Selective ratio $|C^{\star}|/|C|$ |
|--------------------------------------------|-------------------------|--------------------|-----------------|-----------------------------------|
| No defense                                 | 0                       | [XX.X]             | [XX.X]          | -                                 |
| SOR / ROR                                  | [XX.X]                  | [XX.X]             | [XX.X]          | -                                 |
| Uniform SPU diffusion                      | [XX.X]                  | [XX.X]             | [XX.X]          | 1.00                              |
| **ASAP (selective, ours)**                 | **[XX.X]**              | [XX.X]             | **[XX.X]**      | **[X.XX]**                        |

The last column reports the fraction of candidate SPUs that actually trigger M3, $|C^{\star}|/|C|$ in the notation of Algorithm 1. Because M1 + M2 are closed-form and per-SPU, the *purification cost* row of ASAP is expected to scale roughly linearly with this selective ratio, while Uniform SPU diffusion always runs M3 on all $|C|$ candidates.
