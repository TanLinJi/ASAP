# 2. Related Work

<!-- 目标长度: 约半栏。ICASSP 4 页里相关工作必须压缩，每类只点 2 - 3 个代表性工作。 -->

## 2.1 Adversarial attacks on LiDAR detection

<!--
- 物理可实现的点注入攻击。
- 点扰动 / 点丢弃攻击。
- 跨检测器迁移性。
-->

Adversarial attacks on LiDAR detectors mainly fall into three families. *Point-injection attacks* place a small number of malicious points to spoof, hide, or shift target objects, and several have been shown to be physically realizable with low-cost retro-reflective surfaces [@cao2019adversarial; @tu2020physically; @sun2020towards]. *Point-perturbation attacks* shift the coordinates of existing returns within a bounded budget to mislead 3D detectors [@xiang2019pointcloud; @liu2019extending], whereas *point-dropping attacks* delete carefully chosen returns to make objects vanish from the prediction set [@zheng2019pointcloud]. Crucially, recent studies report that adversarial perturbations crafted against one detector often transfer to others trained on the same dataset, motivating defenses that do **not** depend on the specific detector being protected.

## 2.2 Defenses for point-cloud detection

<!--
- 训练时鲁棒化：adversarial training, augmentation。
- 检测器内嵌过滤：learned outlier removal。
- 通用统计去噪：SOR, ROR, voxel denoising。
- 强调：大多需要重训或损害干净精度。
-->

Defenses for point-cloud detectors broadly cluster into three families, each with practical limitations. *Adversarial training and data augmentation* improve robustness by re-fitting detector weights on perturbed scans [@liu2019extending; @sun2020towards], but they are detector-specific and need to be repeated whenever the architecture or training schedule changes. *Learned outlier-removal modules* embedded inside the detector backbone can suppress some malicious returns [TODO: cite-PointGuard], yet they entangle defense with detector internals and offer no guarantees when reused on a different model. *Classical statistical denoisers* such as Statistical Outlier Removal (SOR) and Radius Outlier Removal (ROR) [@rusu2008towards] are training-free and architecture-agnostic, but apply the same isotropic smoothing across the whole scan, often removing legitimate sparse returns from distant objects and degrading clean-scene detection accuracy. ASAP shares the training-free and detector-agnostic spirit of statistical denoisers, but replaces global smoothing with a *localized, anomaly-guided* purification step.

## 2.3 Diffusion-based point-cloud purification

<!--
- PointDP / Ada3Diff 等扩散净化方法。
- 优点：模型无关、概念清晰。
- 限制：scene-wide 扩散，对 LiDAR 稀疏点云成本高，会破坏干净几何。
- 我们的差异：local + selective + adaptive radius。
-->

Diffusion-based purification has recently emerged as a strong, model-agnostic defense paradigm in the image domain [@nie2022diffpure], and has been extended to 3D point clouds through pretrained score-based generative models [@sun2023ada3diff; @luo2021diffusion]. These methods project the corrupted input back toward the clean data manifold by running a short forward-then-reverse Stochastic Differential Equation (SDE) trajectory [@song2021scorebased], and they require no detector retraining. Applied directly to a full LiDAR scan, however, they are *scene-wide*: the diffusion model is evaluated on tens of thousands of points per frame, which is costly, and the indiscriminate noising step also blurs benign object geometry. ASAP differs from prior diffusion purifiers in three ways: (i) it is **local**, operating inside per-point Spherical Purification Units rather than over the entire scene; (ii) it is **selective**, invoking the diffusion model only on units flagged as suspicious by the M2 anomaly scorer, leaving low-risk regions strictly untouched; and (iii) it is **adaptive**, with SPU radii that follow local point density so that both nearby dense surfaces and far-away sparse returns are purified at the right spatial scale.
