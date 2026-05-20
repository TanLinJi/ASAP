# 2. Related Work

<!-- 目标长度: 约半栏。ICASSP 4 页里相关工作必须压缩，每类只点 2 - 3 个代表性工作。 -->

## 2.1 Adversarial attacks on LiDAR detection

<!--
- 物理可实现的点注入攻击。
- 点扰动 / 点丢弃攻击。
- 跨检测器迁移性。
-->

TODO: 2 - 3 sentences. Cite representative LiDAR detection attack works covering point injection, perturbation, and dropping. Highlight that these attacks transfer across detectors.

## 2.2 Defenses for point-cloud detection

<!--
- 训练时鲁棒化（adversarial training, augmentation）。
- 检测器内嵌过滤（learned outlier removal）。
- 通用统计去噪（SOR, ROR, voxel denoising）。
- 强调：大多需要重训或损害干净精度。
-->

TODO: 3 - 4 sentences covering adversarial training, learned outlier removal, and statistical denoising. Emphasize that these defenses either need retraining or degrade clean accuracy.

## 2.3 Diffusion-based point-cloud purification

<!--
- PointDP / Ada3Diff 等扩散净化方法。
- 优点：模型无关、概念清晰。
- 限制：scene-wide 扩散，对 LiDAR 稀疏点云成本高，会破坏干净几何。
- 我们的差异：local + selective + adaptive radius。
-->

TODO: 3 - 4 sentences positioning ASAP against scene-wide diffusion purifiers. Make the differences explicit: local (per-SPU), selective (only suspicious SPUs), and adaptive (density-aware radius).
