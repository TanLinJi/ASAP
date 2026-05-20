# 1. Introduction

<!-- 目标长度: ICASSP 双栏第一栏约 3/4 长度，含 1 张可选示意图。 -->

## 1.1 Background and motivation

<!--
要点（first-draft 草稿前的写作蓝图）：
- LiDAR 3D 目标检测是自动驾驶感知核心。
- 近年攻击工作表明检测器对点云对抗攻击（injection / perturbation / dropping）非常脆弱。
- 攻击方式简单且物理可实现，造成现实安全风险。
-->

LiDAR-based 3D object detection has become a core perception module of modern autonomous driving systems, supporting downstream prediction, planning, and control [@geiger2012kitti; @sun2020waymo; @caesar2020nuscenes]. Recent work, however, shows that point-cloud detectors are highly vulnerable to *adversarial attacks* that inject a small number of points [@cao2019adversarial; @tu2020physically], perturb existing point coordinates [@xiang2019pointcloud], or drop carefully chosen returns [@zheng2019pointcloud]. Because LiDAR sensors operate in an open physical environment, several of these attacks are realizable in the real world and transfer across detector architectures, making the security of LiDAR-based perception a pressing concern.

## 1.2 Limitations of existing defenses

<!--
要点：
- detector-coupled：依赖检测器、需要重训。
- 统一去噪：把干净几何也抹掉，clean accuracy 掉。
- 现有点云扩散净化：scene-wide，成本高、对稀疏 LiDAR 不友好。
- LiDAR-SPD 已经引入球形单元思想，但半径固定且没有显式异常评分。
-->

Most existing defenses for point-cloud detection fall into one of three groups, each with significant drawbacks for deployment. *Detector-coupled* defenses, such as adversarial training and learned outlier removal heads, must be re-applied for every new detector and typically require costly retraining [@liu2019extending; @sun2020towards]. *Uniform statistical denoisers*, such as Statistical Outlier Removal (SOR) and Radius Outlier Removal (ROR) [@rusu2008towards], apply the same smoothing rule everywhere; while they remove some adversarial points, they also strip away legitimate sparse returns from distant objects and harm clean detection accuracy. *Scene-wide diffusion purifiers* recently proposed for point clouds [@sun2023ada3diff; @nie2022diffpure] are detector-agnostic but run a generative model over the entire scene, which is expensive on dense LiDAR scans and tends to over-smooth benign object geometry. A complementary line of work [@lidarspd] processes points inside local spherical units, but uses a fixed radius and lacks an explicit anomaly score to decide *where* purification is actually needed.

## 1.3 Our approach

<!--
- ASAP = detector-agnostic、inference-only、selective purification。
- M1 Adaptive Radius / M2 Anomaly Scorer / M3 Selective Diffusion。
- 一句话总结：只对"看起来异常"的局部球形区域做扩散净化，其余原样保留。
-->

We propose **ASAP** (**A**daptive **S**pherical **A**nomaly-Guided **P**urification), a detector-agnostic and inference-only purification framework that addresses these three limitations jointly. Given an input LiDAR scan that may contain adversarial points, ASAP first decomposes the scene into **Spherical Purification Units (SPUs)** whose radii are *adapted to local point density* (**M1**), so that both nearby dense surfaces and far-away sparse returns are analyzed at appropriate spatial scales. Each SPU is then assigned an **anomaly score** $s(p) \in [0, 1]$ via a lightweight, label-free combination of four geometric statistics — compactness, PCA-based anisotropy, von Mises-Fisher angular concentration, and local density ratio (**M2**) — whose joint behavior is hard for benign road objects to violate. Only SPUs with $s(p) > \tau$ are passed through a Variance-Preserving Stochastic Differential Equation (VP-SDE) point-cloud purifier [@song2021scorebased] (**M3**) on a short reverse trajectory, while low-risk regions are returned verbatim. The purified scan is consumed by any off-the-shelf 3D detector without architectural or weight modification, making ASAP a drop-in front-end for the detector pipeline.

## 1.4 Contributions

We summarize our contributions as follows:

- **C1 (Framework).** We propose ASAP, the first detector-agnostic and inference-only LiDAR purification framework that *selectively* applies diffusion-based purification at the level of local spherical units rather than scene-wide, decoupling defense from detector training.
- **C2 (Adaptive locality).** We introduce a density-adaptive radius rule based on $k$-NN local scale estimation that decomposes a LiDAR scan into spherical purification units whose statistical reliability is guaranteed under a locally Poisson assumption.
- **C3 (Label-free anomaly scoring).** We design a lightweight geometric anomaly scorer that aggregates four closed-form features (compactness, PCA anisotropy, von Mises-Fisher concentration, density ratio) and calibrates a decision threshold via ROC + Youden-$J$ [@youden1950index], all without requiring any adversarial labels for the detector under attack.
- **C4 (Empirical evidence).** Across multiple 3D detector architectures on **KITTI** (sanity check) and **Waymo / nuScenes** (modern detectors such as DSVT, VoxelNeXt, MPPNet, TransFusion-Lidar), ASAP mitigates representative LiDAR adversarial attacks — reducing the attack success rate by [XX.X%] while preserving [YY.Y%] of clean detection performance — without retraining any detector. *(Numerical claims to be filled after experiment milestones MS2-MS3.)*

<!-- 投稿前清理 TODO/注释；contributions 数量按最终情况收敛为 3 条或 4 条。 -->
