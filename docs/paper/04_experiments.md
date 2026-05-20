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

TODO: write the setup section in 4 - 6 sentences.

- **Datasets**: KITTI is used for fast reproducibility and ablation studies, while Waymo Open Dataset and nuScenes are used for the main detector-agnostic evaluation with recent strong detectors.
- **KITTI detectors**: PointPillars and PV-RCNN++ are used as controlled baselines because KITTI is convenient for debugging and comparison with prior defenses.
- **Waymo detectors**: DSVT-Voxel, VoxelNeXt, and MPPNet are the primary modern detector backbones, covering dynamic sparse voxel transformers, fully sparse voxel detectors, and temporal multi-frame detectors.
- **nuScenes detectors**: VoxelNeXt and TransFusion-Lidar are used to test whether ASAP transfers to a different large-scale benchmark and a different detection head.
- **Metrics**: KITTI reports 3D Average Precision (AP); Waymo reports mAP / mAPH; nuScenes reports mAP and NDS.
- **Attacks**: at least one representative attack per category (injection, perturbation, dropping).
- **Hardware**: early KITTI development runs on the local CUDA 12.4 setup; large-scale Waymo / nuScenes experiments may require stronger GPUs or official pre-trained checkpoints.

## 4.2 Baselines

<!--
- No defense（原始攻击下检测）。
- 经典统计去噪：SOR / ROR。
- Uniform SPU diffusion (ours, no gating)：作为公平消融，证明 selective gating 是关键。
- 可选：再加一种 published purification 作为外部对照（按字数允许补充）。
-->

We compare ASAP against the following baselines:

- **No defense**: the attacked scan is fed directly to the detector.
- **SOR / ROR**: statistical and radius outlier removal on the full scan.
- **Uniform SPU diffusion** (our own ablation, no gating): diffuses every SPU, isolating the contribution of the anomaly-guided selection.
- *(Optional)* one published diffusion purifier for external reference.

## 4.3 Main results

<!--
- 表 1：主结果不再只放 KITTI/PointPillars，而是以 detector 为行。
- 建议列: Dataset, Detector, Attack, No defense, Uniform SPU, ASAP, Clean upper bound。
- KITTI 详细 Car/Pedestrian/Cyclist AP 可以放补充表或附表。
- Waymo / nuScenes 用 mAP/mAPH/NDS 支撑 modern detector-agnostic claim。
-->

TODO: insert main results table.

| Dataset | Detector | Detector family | Attack | Clean | No defense | Uniform SPU | **ASAP (ours)** |
|---------|----------|-----------------|--------|-------|------------|-------------|-----------------|
| KITTI | PointPillars | Pillar-based one-stage | Injection | TBD AP | TBD AP | TBD AP | TBD AP |
| KITTI | PV-RCNN++ | Voxel-point two-stage | Injection | TBD AP | TBD AP | TBD AP | TBD AP |
| Waymo | DSVT-Voxel | Dynamic sparse voxel transformer | Injection | TBD mAP/mAPH | TBD mAP/mAPH | TBD mAP/mAPH | TBD mAP/mAPH |
| Waymo | VoxelNeXt | Fully sparse voxel detector | Injection | TBD mAP/mAPH | TBD mAP/mAPH | TBD mAP/mAPH | TBD mAP/mAPH |
| Waymo | MPPNet | Temporal multi-frame detector | Injection | TBD mAP/mAPH | TBD mAP/mAPH | TBD mAP/mAPH | TBD mAP/mAPH |
| nuScenes | VoxelNeXt | Fully sparse voxel detector | Injection | TBD mAP/NDS | TBD mAP/NDS | TBD mAP/NDS | TBD mAP/NDS |
| nuScenes | TransFusion-Lidar | Transformer/query detection head | Injection | TBD mAP/NDS | TBD mAP/NDS | TBD mAP/NDS | TBD mAP/NDS |

## 4.4 Ablation study

<!--
- A1: 固定半径 vs M1 自适应半径。
- A2: 单一异常特征 vs M2 全部 4 个特征组合。
- A3: 全部扩散 vs M3 阈值控制选择性扩散。
- A4: 阈值 tau 的敏感性曲线。
-->

We ablate the three modules of ASAP independently:

- **A1.** Adaptive vs fixed SPU radius.
- **A2.** Full anomaly scorer vs each single feature.
- **A3.** Selective diffusion vs uniform diffusion.
- **A4.** Sensitivity to the anomaly threshold $\tau$ along its ROC curve.

TODO: insert the ablation table once experiments land.

## 4.5 Inference cost

<!--
- 报告：clean 推理时间 / 受扰未净化时间 / 全扩散净化时间 / ASAP 选择性净化时间。
- 单位 ms per frame，单卡 GPU。
- 强调 selective gating 的加速倍率。
-->

TODO: report wall-clock inference cost per frame for no defense, uniform diffusion, and ASAP. Emphasize the speedup brought by selective gating.
