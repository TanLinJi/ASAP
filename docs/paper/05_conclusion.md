# 5. Conclusion

<!-- 目标长度: 约 1/4 栏，4 - 6 句。 -->

<!--
要点：
- 重述问题：LiDAR 检测器对点云对抗攻击脆弱。
- 重述方法：ASAP 用自适应球形单元 + 几何异常评分 + 选择性扩散。
- 重述优点：detector-agnostic、inference-only、避免破坏干净几何、计算成本低。
- 展望：扩展到 nuScenes / Waymo；扩展到更强自适应攻击；与训练时鲁棒化结合。
-->

TODO: write the final conclusion paragraph once the experiments converge. Outline:

1. Restate the threat: LiDAR detectors are vulnerable to adversarial point-cloud attacks.
2. Restate the method: ASAP combines an adaptive radius rule, a lightweight geometric anomaly scorer, and selective VP-SDE diffusion at the level of spherical purification units.
3. Restate the empirical message: ASAP recovers most of the clean detection accuracy in a detector-agnostic, inference-only manner, with a controlled compute cost.
4. Mention one or two future directions: extension to nuScenes / Waymo, evaluation against stronger adaptive attacks, and combination with training-time robustification.
