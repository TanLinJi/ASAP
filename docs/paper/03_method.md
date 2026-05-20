# 3. ASAP Framework

<!-- 目标长度: 1.5 - 2 栏，含 1 张架构图（建议放在 3.1 末尾）。 -->

## 3.1 Overview

<!--
- 一段话总结流水线：输入受扰点云 P -> 候选 SPU 选择 -> M1 自适应半径 -> M2 异常评分 -> M3 选择性扩散 -> 输出净化点云 P_hat -> 任意 LiDAR 检测器。
- 强调 detector-agnostic 与 inference-only。
- 在末尾引用一张 pipeline 图: Fig. 1。
-->

Let $P = \{p_i\}_{i=1}^{N} \subset \mathbb{R}^3$ be an input LiDAR scan that may contain adversarial points. ASAP returns a purified scan $\hat{P}$ in three stages:
**(1) Adaptive Radius (M1)** builds a set of Spherical Purification Units (SPUs) whose radii adapt to local density,
**(2) Anomaly Scorer (M2)** assigns a lightweight geometric score to each SPU,
**(3) Selective Diffusion (M3)** runs a VP-SDE point-cloud purifier only on SPUs whose score exceeds a calibrated threshold $\tau$, leaving benign regions intact.
The purified scan $\hat{P}$ is fed to any off-the-shelf LiDAR detector without modification.

> Fig. 1: ASAP pipeline overview. *(figure to be added)*

## 3.2 Adaptive Radius (M1)

<!--
- 公式（最小可写版本，定稿时再换 LaTeX）：
  对点 p，取其 k 近邻 d_k(p)，定义 r1(p) = clip( alpha * d_k(p), [r_min, r_max] )。
- 默认参数：alpha = 2, k = 16, [r_min, r_max] = [0.10 m, 0.40 m]，r2 = 0.67 * r1。
- 写清楚 r1 是外圈半径，r2 是内圈半径；外圈点用于评分上下文，内圈点是真正会被扩散的点。
-->

For each candidate center $p$, let $d_k(p)$ denote the distance to its $k$-th nearest neighbor in $P$. We define the outer SPU radius as
$$
r_1(p) = \mathrm{clip}\big(\alpha\, d_k(p),\, [r_{\min},\, r_{\max}]\big),
$$
with default $\alpha = 2$, $k = 16$, $r_{\min} = 0.10$ m, and $r_{\max} = 0.40$ m. The inner radius for purification is $r_2(p) = 0.67\, r_1(p)$. Points within $r_2(p)$ are candidates for diffusion, while points within the annulus $[r_2(p), r_1(p)]$ provide context for the anomaly scorer.

## 3.3 Anomaly Scorer (M2)

<!--
- 四个轻量统计特征：
  1) compactness：内圈点数 / 球体积。
  2) anisotropy：PCA 特征值比例，例如 lambda_3 / (lambda_1 + lambda_2 + lambda_3)。
  3) vMF kappa：把归一化方向向量套到 von Mises-Fisher 上的浓度估计。
  4) density：内外圈密度比 rho_inner / rho_outer。
- 用线性组合或 logistic 模型给出 score s(p) ∈ [0, 1]。
- 阈值 tau 用 ROC 标定：在干净/受扰 SPU 上取最优 Youden-J 阈值。
-->

We compute four scalar features per SPU centered at $p$:

- **Compactness** $f_c(p)$: ratio of the inner point count to the inner volume.
- **PCA anisotropy** $f_a(p)$: based on the smallest-to-total eigenvalue ratio of the covariance of inner points.
- **vMF concentration** $f_v(p)$: maximum-likelihood concentration $\kappa$ when fitting a von Mises-Fisher distribution to the unit direction vectors $\frac{p_j - p}{\lVert p_j - p \rVert}$.
- **Density ratio** $f_d(p)$: ratio between inner and annulus densities.

These features are combined into a single anomaly score $s(p) \in [0, 1]$ via a simple monotone aggregator (linear combination or a small logistic model). An SPU is marked suspicious when $s(p) > \tau$, where $\tau$ is calibrated on a held-out clean/attacked SPU set using the Youden-J statistic on the ROC curve.

## 3.4 Selective Diffusion (M3)

<!--
- 只对 s(p) > tau 的 SPU 做扩散：把内圈点 P_inner 送进 VP-SDE 风格的点云扩散模型。
- 写清楚扩散仅作用于局部 SPU，不是整场景。
- 多个可疑 SPU 用 padding + 批处理加速。
- 输出点云：把扩散后的内圈点替换原内圈点，外圈点保持不变。
-->

For each suspicious SPU, the inner points $P_{\mathrm{inner}}(p) = \{p_j : \lVert p_j - p \rVert \le r_2(p)\}$ are passed to a VP-SDE point-cloud diffusion module that performs a short forward-then-reverse trajectory in coordinate space. The purified inner points $\hat{P}_{\mathrm{inner}}(p)$ replace the original inner points in $P$, while points outside any suspicious SPU stay untouched. Multiple suspicious SPUs are padded to a common point count and processed in one batch, so the total purification cost scales with the number of suspicious SPUs rather than with the full point-cloud size.

## 3.5 Practical considerations

<!--
- 复杂度分析：KNN 是主要开销，可用 GPU 加速。
- 不需要 KITTI 训练 label，纯几何方法。
- 与现有检测器（PointPillars / PV-RCNN / CenterPoint）兼容。
-->

TODO: 2 - 3 sentences on overall complexity, memory footprint, and detector compatibility.
