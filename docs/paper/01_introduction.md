# 1. Introduction

<!-- 目标长度: ICASSP 双栏第一栏约 3/4 长度，含 1 张可选示意图。 -->

## 1.1 Background and motivation

<!--
要点（first-draft 草稿前的写作蓝图）：
- LiDAR 3D 目标检测是自动驾驶感知核心。
- 近年攻击工作表明检测器对点云对抗攻击（injection / perturbation / dropping）非常脆弱。
- 攻击方式简单且物理可实现，造成现实安全风险。
-->

TODO: write 2 - 3 sentences on the role of LiDAR detection in autonomous driving and the security risks of point-cloud adversarial attacks. Cite representative attack works such as physical point injection and perturbation methods.

## 1.2 Limitations of existing defenses

<!--
要点：
- 检测器内嵌防御：依赖检测器结构、需重训。
- 统一去噪/平滑：会把干净几何也抹掉，导致干净精度大幅下降。
- 现有点云扩散净化：成本高、对 LiDAR 稀疏点云结构不够友好。
- 既有 LiDAR-SPD 思路：球形单元能定位扰动，但半径固定、未结合异常分。
-->

TODO: list 3 limitations of prior defenses in 3 - 4 sentences:
1. detector-coupled or retraining-required defenses;
2. uniform denoising or smoothing that hurts clean accuracy;
3. existing point-cloud diffusion purifiers applied scene-wide that are expensive and over-smooth benign structure.

## 1.3 Our approach

<!--
要点：
- ASAP 是 detector-agnostic、inference-only、selective 的 purification 框架。
- 三个模块：M1 Adaptive Radius / M2 Anomaly Scorer / M3 Selective Diffusion。
- 思想总结一句：只对"看起来异常"的局部球形区域做扩散净化，其余原样保留。
-->

TODO: introduce ASAP in 4 - 5 sentences, named after **Adaptive Spherical Anomaly-Guided Purification**, with the three modules **Adaptive Radius (M1)**, **Anomaly Scorer (M2)**, and **Selective Diffusion (M3)**. Stress the detector-agnostic and inference-only properties.

## 1.4 Contributions

We summarize our contributions as follows:

- **C1.** We propose ASAP, the first detector-agnostic, inference-only LiDAR purification framework that *selectively* applies diffusion-based purification at the level of local spherical units rather than scene-wide.
- **C2.** We design a density-adaptive radius rule and a lightweight geometric anomaly scorer (compactness, PCA anisotropy, von Mises-Fisher concentration, density) that together identify suspicious spherical purification units with negligible compute.
- **C3.** We integrate VP-SDE point-cloud diffusion as a selective purifier triggered only by the anomaly scorer, which avoids over-smoothing benign structure and reduces inference cost.
- **C4.** On the KITTI 3D detection benchmark we show that ASAP recovers detection accuracy under representative point-cloud attacks while remaining detector-agnostic. *(Numerical claims to be filled.)*

<!-- 投稿前清理 TODO/注释；contributions 数量按最终情况收敛为 3 条或 4 条。 -->
