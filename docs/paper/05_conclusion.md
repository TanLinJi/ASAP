# 5. Conclusion

<!-- 目标长度: 约 1/4 栏，4 - 6 句。 -->

<!--
要点：
- 重述问题：LiDAR 检测器对点云对抗攻击脆弱。
- 重述方法：ASAP 用自适应球形单元 + 几何异常评分 + 选择性扩散。
- 重述优点：detector-agnostic、inference-only、避免破坏干净几何、计算成本低。
- 展望：扩展到 nuScenes / Waymo；扩展到更强自适应攻击；与训练时鲁棒化结合。
-->

We presented **ASAP**, an Adaptive Spherical Anomaly-Guided Purification framework that defends LiDAR-based 3D object detectors against adversarial point-cloud attacks without retraining any detector. ASAP decomposes a LiDAR scan into Spherical Purification Units whose radii adapt to local point density (**M1**), assigns each unit a label-free geometric anomaly score by combining compactness, PCA-based anisotropy, von Mises-Fisher concentration, and density ratio (**M2**), and applies a short-trajectory VP-SDE point-cloud purifier only to the units flagged as suspicious (**M3**). This *local, selective, and adaptive* design lets ASAP suppress localized adversarial perturbations while leaving benign geometry untouched, in stark contrast to scene-wide diffusion purifiers and uniform statistical denoisers. Experiments across KITTI, Waymo, and nuScenes with both classical (PointPillars, PV-RCNN) and recent strong detectors (DSVT, VoxelNeXt, MPPNet, TransFusion-Lidar) show that ASAP reduces the attack success rate by **[XX.X%]** while preserving **[YY.Y%]** of clean detection performance, in a fully inference-only and detector-agnostic manner. *(Numerical claims will be filled in after experiment milestones MS2-MS3.)* Promising future directions include adaptive attacks aware of ASAP's gating, combination with training-time robustification, and extension of the SPU scoring rule to multi-modal LiDAR-camera detectors.
