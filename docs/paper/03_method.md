# 3. ASAP Framework

<!-- 目标长度: 1.5 - 2 栏 + 1 张 pipeline 图 + 1 个 algorithm box。论文行文用英文；HTML 注释保留中文备忘。 -->

## 3.1 Overview

<!--
- 一段话总结流水线：输入受扰点云 P -> 候选 SPU 选择 -> M1 自适应半径 -> M2 异常评分 -> M3 选择性扩散 -> 输出净化点云 P_hat -> 任意 LiDAR 检测器。
- 强调 detector-agnostic 与 inference-only。
- 在末尾引用 Fig. 1。
-->

Let $P = \{p_i\}_{i=1}^{N} \subset \mathbb{R}^3$ be an input LiDAR scan that may contain adversarial points. ASAP returns a purified scan $\hat{P}$ that is fed, without any modification, to an off-the-shelf 3D detector $\mathcal{D}$. ASAP is built around three modules:

- **M1 — Adaptive Radius.** Decompose the scene into a collection of Spherical Purification Units (SPUs) whose radii adapt to local point density, so that sparse and dense regions are analyzed at appropriate spatial scales.
- **M2 — Anomaly Scorer.** Assign each SPU a scalar anomaly score $s(p) \in [0, 1]$ based on four lightweight, label-free geometric statistics whose joint behavior is hard to satisfy by benign objects but typical for adversarial point structures.
- **M3 — Selective Diffusion.** Apply a Variance-Preserving Stochastic Differential Equation (VP-SDE) point-cloud purifier only to SPUs whose score exceeds a Receiver Operating Characteristic (ROC)-calibrated threshold $\tau$, while keeping low-risk regions strictly unchanged.

ASAP is **detector-agnostic** because $\hat{P}$ is still a raw LiDAR point cloud; it is **inference-only** because no detector parameter is touched. Figure 1 illustrates the pipeline.

> Fig. 1: ASAP pipeline overview. *(figure to be added)*

## 3.2 Notation

<!-- 把整章符号集中到一处，方便读者查阅，也便于代码与公式一一对应。 -->

| Symbol | Meaning |
|--------|---------|
| $P = \{p_i\}_{i=1}^{N}$ | Input LiDAR point cloud, $p_i \in \mathbb{R}^3$. |
| $\hat{P}$ | Purified point cloud returned by ASAP. |
| $\mathcal{D}$ | Any off-the-shelf 3D detector. |
| $p$ | Candidate SPU center. |
| $p_{(k)}(p)$ | The $k$-th nearest neighbor of $p$ in $P \setminus \{p\}$. |
| $d_k(p)$ | Distance from $p$ to $p_{(k)}(p)$. |
| $r_1(p),\, r_2(p)$ | Outer and inner radii of the SPU centered at $p$. |
| $\mathcal{S}(p)$ | Full SPU: $\{q \in P : \lVert q - p \rVert \le r_1(p)\}$. |
| $P_{\mathrm{in}}(p)$ | Inner points: $\{q \in P : \lVert q - p \rVert \le r_2(p)\}$. |
| $P_{\mathrm{an}}(p)$ | Annulus points: $\mathcal{S}(p) \setminus P_{\mathrm{in}}(p)$. |
| $f_c, f_a, f_v, f_d$ | Compactness, anisotropy, vMF concentration, density-ratio features. |
| $s(p) \in [0, 1]$ | Aggregated anomaly score. |
| $\tau$ | ROC-calibrated decision threshold. |
| $\beta(t),\, \bar{\alpha}_t$ | VP-SDE noise schedule and survival coefficient. |
| $s_\theta(\cdot, t)$ | Pretrained point-cloud score network. |

## 3.3 Adaptive Radius (M1)

<!--
- 数学动机：LiDAR 远近密度差极大；固定半径要么近处包过整个物体，要么远处球内点太少；统计估计需要"球内点数 ~ 常数"。
- 关键公式：
    r1(p) = clip(alpha * d_k(p), [r_min, r_max])
    r2(p) = beta * r1(p)
- 默认: alpha = 2, k = 16, beta = 0.67, [r_min, r_max] = [0.10 m, 0.40 m]。
-->

### 3.3.1 Local scale from $k$-nearest-neighbor distance

LiDAR scans exhibit highly non-uniform density: nearby surfaces are sampled densely while distant regions are sampled sparsely. A fixed-radius decomposition would either oversmooth dense areas or starve sparse areas of statistics. We therefore tie the SPU radius to a local scale estimated from the data itself.

For a candidate center $p$, let $p_{(k)}(p)$ be its $k$-th nearest neighbor in $P \setminus \{p\}$ and define the local scale
$$
d_k(p) \;=\; \lVert p - p_{(k)}(p) \rVert_2.
$$

Under a locally Poisson assumption with intensity $\rho(p)$, the expected number of points inside a ball of radius $r$ centered at $p$ equals $\frac{4}{3}\pi r^3 \rho(p)$. Setting this equal to $k$ at $r = d_k(p)$ gives
$$
\rho(p) \;\approx\; \frac{k}{\tfrac{4}{3}\pi\, d_k(p)^3} \quad\Longleftrightarrow\quad d_k(p) \;\propto\; \rho(p)^{-1/3}.
$$

Hence $d_k(p)$ is a robust, scale-equivariant estimator of the average inter-point distance around $p$.

### 3.3.2 Outer and inner radii

We define the outer SPU radius
$$
r_1(p) \;=\; \mathrm{clip}\!\big(\alpha\, d_k(p),\; [\,r_{\min},\, r_{\max}\,]\big),
\qquad \alpha > 1,
\tag{M1.1}
$$
and the inner radius
$$
r_2(p) \;=\; \beta\, r_1(p),\qquad \beta \in (0, 1).
\tag{M1.2}
$$

The clipping serves two purposes:

- **Upper bound $r_{\max}$** prevents the SPU from spanning across multiple distinct objects in extremely sparse regions.
- **Lower bound $r_{\min}$** prevents statistically unreliable SPUs in oversampled regions where $d_k(p) \to 0$.

Choosing $\alpha > 1$ guarantees that the expected number of points inside the outer ball grows at least linearly in $k$ under Poisson sampling, namely
$$
\mathbb{E}\big[|\mathcal{S}(p)|\big] \;\ge\; \alpha^3 k,
$$
which keeps every SPU statistically well-populated.

### 3.3.3 SPU definition and decomposition

An SPU centered at $p$ is the triple $\big(\mathcal{S}(p),\, P_{\mathrm{in}}(p),\, P_{\mathrm{an}}(p)\big)$ with
$$
\mathcal{S}(p) \;=\; \{q \in P : \lVert q - p\rVert_2 \le r_1(p)\},\qquad
P_{\mathrm{in}}(p) \;=\; \{q \in P : \lVert q - p\rVert_2 \le r_2(p)\},
$$
$$
P_{\mathrm{an}}(p) \;=\; \mathcal{S}(p) \setminus P_{\mathrm{in}}(p).
$$

The inner points $P_{\mathrm{in}}$ are the only ones eligible for diffusion (M3), while the annulus points $P_{\mathrm{an}}$ stabilize the anomaly statistics in M2 and never get edited.

To avoid redundant SPUs, candidate centers are drawn from $P$ via Farthest Point Sampling (FPS) at a sampling stride $\eta$. Let $C \subset P$ denote the set of selected centers. With proper $\eta$ relative to $r_1$, the union $\bigcup_{p \in C} \mathcal{S}(p)$ provides an $r_1$-cover of $P$ with bounded overlap.

### 3.3.4 Default hyperparameters

We use the defaults
$$
\alpha = 2,\quad k = 16,\quad \beta = 0.67,\quad r_{\min} = 0.10\text{ m},\quad r_{\max} = 0.40\text{ m},
$$
which empirically place an SPU at the scale of a small pedestrian limb or vehicle wheel — the smallest spatial unit on which adversarial perturbations remain meaningful for KITTI/Waymo class definitions.

## 3.4 Anomaly Scorer (M2)

<!--
- 4 个特征：compactness、PCA anisotropy、vMF kappa、density ratio。
- 都是闭式估计、无需训练。
- 用 logistic 聚合得到 s(p) ∈ [0,1]。
- 阈值 τ 由 ROC + Youden-J 在干净/受扰 SPU 上标定。
-->

The anomaly scorer summarizes each SPU through four label-free geometric statistics and aggregates them into a single anomaly score $s(p) \in [0, 1]$. Each feature is chosen so that adversarial point structures (clusters, planar injections, directional sprays, density anomalies) tend to violate at least one statistic, while benign LiDAR returns from common road objects rarely violate all four simultaneously.

### 3.4.1 Compactness $f_c$

Define the inner volume $V_2(p) = \tfrac{4}{3}\pi\, r_2(p)^3$. The compactness feature is the inner-ball point density:
$$
f_c(p) \;=\; \frac{|P_{\mathrm{in}}(p)|}{V_2(p)}.
\tag{M2.1}
$$

Adversarial injection attacks typically place a tight cluster of points to mimic an object and therefore induce abnormally high $f_c$.

### 3.4.2 PCA anisotropy $f_a$

Let $P_{\mathrm{in}}(p) = \{q_1, \dots, q_m\}$ with centroid $\bar{q} = \tfrac{1}{m}\sum_{j=1}^{m} q_j$. The empirical covariance is
$$
\Sigma(p) \;=\; \frac{1}{m}\sum_{j=1}^{m} (q_j - \bar{q})(q_j - \bar{q})^{\top} \;\in\; \mathbb{R}^{3 \times 3},
$$
with eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge 0$. The anisotropy feature is
$$
f_a(p) \;=\; 1 - \frac{\lambda_3}{\lambda_1 + \lambda_2 + \lambda_3}.
\tag{M2.2}
$$

Limiting cases:

- **Isotropic blob:** $\lambda_1 \approx \lambda_2 \approx \lambda_3 \;\Rightarrow\; f_a \approx 2/3$.
- **Planar surface:** $\lambda_3 \approx 0 \;\Rightarrow\; f_a \approx 1$.
- **Linear structure:** $\lambda_2 \approx \lambda_3 \approx 0 \;\Rightarrow\; f_a \approx 1$.

Adversarial sprays often align along a sensor-aware direction and hence produce locally low-rank patches with $f_a \to 1$.

### 3.4.3 vMF angular concentration $f_v$

For each inner point $q_j$, compute the unit direction relative to the SPU center
$$
u_j \;=\; \frac{q_j - p}{\lVert q_j - p\rVert_2} \;\in\; \mathbb{S}^2.
$$
We model $\{u_j\}_{j=1}^{m}$ as samples from a 3D von Mises-Fisher (vMF) distribution
$$
p(u; \mu, \kappa) \;=\; C_3(\kappa)\, \exp(\kappa\, \mu^{\top} u),\qquad \mu \in \mathbb{S}^2,\ \kappa \ge 0,
$$
with normalization $C_3(\kappa) = \tfrac{\kappa}{4\pi \sinh \kappa}$.

Let $\bar{u} = \tfrac{1}{m}\sum_{j=1}^{m} u_j$ and $\bar{R} = \lVert \bar{u} \rVert_2 \in [0, 1]$. We use the closed-form approximation of the maximum-likelihood concentration estimator from [@banerjee2005clustering]:
$$
\hat{\kappa}(p) \;=\; \frac{\bar{R}\,(d - \bar{R}^{2})}{1 - \bar{R}^{2}},\qquad d = 3.
\tag{M2.3}
$$

The vMF feature is
$$
f_v(p) \;=\; \hat{\kappa}(p).
$$

Higher $\hat{\kappa}$ means the directions $\{u_j\}$ are tightly concentrated around a single $\mu$. Adversarial point injections that arrive from a narrow angular cone produce high $f_v$, while diffuse benign returns produce low $f_v$.

### 3.4.4 Density ratio $f_d$

Let $V_1(p) = \tfrac{4}{3}\pi r_1(p)^3$ and $V_{\mathrm{an}}(p) = V_1(p) - V_2(p)$. The density ratio is
$$
f_d(p) \;=\; \frac{|P_{\mathrm{in}}(p)| / V_2(p)}{|P_{\mathrm{an}}(p)| / V_{\mathrm{an}}(p) + \varepsilon},
\tag{M2.4}
$$
with a small $\varepsilon > 0$ for numerical stability. Adversarial clusters break the smooth density transition between core and surrounding region, leading to $f_d \gg 1$.

### 3.4.5 Score aggregation

Each raw feature is normalized into a signed deviation using statistics estimated on a clean reference set $P^{\text{clean}}$:
$$
\tilde{f}_\ast(p) \;=\; \frac{f_\ast(p) - \mu_\ast}{\sigma_\ast},\qquad \ast \in \{c, a, v, d\}.
$$

The aggregated anomaly score is a logistic combination
$$
s(p) \;=\; \sigma\!\left(\, w_c\,\tilde{f}_c(p) + w_a\,\tilde{f}_a(p) + w_v\,\tilde{f}_v(p) + w_d\,\tilde{f}_d(p) + b\,\right),
\tag{M2.5}
$$
where $\sigma(x) = (1 + e^{-x})^{-1}$ is the logistic function. The weights $\{w_c, w_a, w_v, w_d, b\}$ are fitted by logistic regression on a small set of labeled SPUs (clean vs. attacked), drawn once and reused across detectors and attacks. Because the four features are themselves analytic and label-free, this fit only adjusts a 5-dimensional aggregator and is essentially independent of any specific detector $\mathcal{D}$.

### 3.4.6 Threshold calibration via ROC + Youden-$J$

The decision rule $D_\tau(p) = \mathbb{1}[s(p) > \tau]$ defines, for any held-out SPU dataset with labels $y(p) \in \{0, 1\}$, a True Positive Rate and a False Positive Rate
$$
\mathrm{TPR}(\tau) = \Pr[D_\tau = 1 \mid y = 1],\qquad \mathrm{FPR}(\tau) = \Pr[D_\tau = 1 \mid y = 0].
$$

Sweeping $\tau \in [0, 1]$ traces the ROC curve. Following the Youden index [@youden1950index], we pick
$$
\tau^{\star} \;=\; \arg\max_{\tau \in [0, 1]}\; J(\tau),\qquad J(\tau) = \mathrm{TPR}(\tau) - \mathrm{FPR}(\tau),
\tag{M2.6}
$$
which maximizes the gap between hits and false alarms under equal cost. Calibration is performed once on a held-out SPU set and the resulting $\tau^{\star}$ is reused at test time.

## 3.5 Selective Diffusion (M3)

<!--
- 用 VP-SDE 描述前向加噪，闭式边际 + 反向 SDE 写成标准 score-based 形式。
- 短轨迹截断：选 t* 让噪声尺度刚好压过对抗扰动幅度。
- 局部应用：只对 s(p) > τ 的 SPU 内点扩散，外圈和未选 SPU 完全不动。
- SPU 重叠时对每个原始点取参与净化 SPU 的平均结果。
-->

### 3.5.1 VP-SDE forward and reverse process

We instantiate M3 with a Variance-Preserving SDE following score-based generative modeling through stochastic differential equations [@song2021scorebased]. For a single 3D coordinate $x \in \mathbb{R}^3$, the forward process is
$$
\mathrm{d}x_t \;=\; -\tfrac{1}{2}\, \beta(t)\, x_t\, \mathrm{d}t \;+\; \sqrt{\beta(t)}\, \mathrm{d}w_t,\qquad t \in [0, T],
\tag{M3.1}
$$
with $\beta(t) > 0$ a noise schedule and $w_t$ a standard Wiener process. Define the survival coefficient
$$
\bar{\alpha}_t \;=\; \exp\!\left(-\int_0^t \beta(s)\, \mathrm{d}s\right).
$$

The marginal at time $t$ admits the closed form
$$
x_t \;=\; \sqrt{\bar{\alpha}_t}\, x_0 \;+\; \sqrt{1 - \bar{\alpha}_t}\, \epsilon,\qquad \epsilon \sim \mathcal{N}(0, I_3),
\tag{M3.2}
$$
which shows that VP-SDE is variance-preserving: if $x_0 \sim \mathcal{N}(0, I_3)$ then $x_t \sim \mathcal{N}(0, I_3)$ for every $t$.

The corresponding reverse-time SDE is
$$
\mathrm{d}x_t \;=\; \left[\,-\tfrac{1}{2}\beta(t)\, x_t \;-\; \beta(t)\, \nabla_{x_t} \log p_t(x_t)\,\right]\mathrm{d}t \;+\; \sqrt{\beta(t)}\, \mathrm{d}\bar{w}_t,
\tag{M3.3}
$$
where $\bar{w}_t$ is a reverse-time Wiener process. We approximate the score $\nabla_{x_t} \log p_t(x_t)$ by a pretrained point-cloud score network $s_\theta(x_t, t)$.

### 3.5.2 Short-trajectory truncation

We do not simulate (M3.3) all the way from $T$ to $0$. Given an attack budget $\delta_{\max}$ that bounds the per-point adversarial displacement, we choose a truncation time $t^{\star} < T$ such that the injected noise scale dominates $\delta_{\max}$ but does not destroy benign geometry:
$$
\sqrt{1 - \bar{\alpha}_{t^{\star}}} \;\approx\; c\, \delta_{\max},\qquad c \in [1, 2].
\tag{M3.4}
$$

This short forward-then-reverse trajectory follows the adversarial purification principle of DiffPure [@nie2022diffpure]. Conceptually, it pushes the local point cloud onto the support of the clean data manifold without traveling all the way to the prior $\mathcal{N}(0, I_3)$.

### 3.5.3 Local application within selected SPUs

For an SPU with $s(p) > \tau^{\star}$, let $X_p \in \mathbb{R}^{m \times 3}$ stack the inner points $P_{\mathrm{in}}(p)$. We apply the following local procedure:

1. **Center and rescale** $X_p$ into a unit ball:
$$
\tilde{X}_p \;=\; \frac{X_p - p}{r_2(p)}.
$$
2. **Forward to $t^{\star}$:** sample $\tilde{X}_p^{(t^{\star})} = \sqrt{\bar{\alpha}_{t^{\star}}}\, \tilde{X}_p + \sqrt{1 - \bar{\alpha}_{t^{\star}}}\, E$ with $E \sim \mathcal{N}(0, I)$, applied row-wise.
3. **Reverse to $0$:** integrate (M3.3) using the pretrained score $s_\theta$ from $t^{\star}$ down to $0$, obtaining $\hat{\tilde{X}}_p$.
4. **Inverse rescale and recenter:**
$$
\hat{X}_p \;=\; r_2(p)\, \hat{\tilde{X}}_p + p.
$$

The purified inner set is $\hat{P}_{\mathrm{in}}(p) = \mathrm{rows}(\hat{X}_p)$. SPUs with $s(p) \le \tau^{\star}$ are skipped entirely; their points are passed through unchanged.

### 3.5.4 Aggregation under SPU overlap

Because SPUs may overlap, a single point $q \in P$ may belong to several inner sets. Let $C(q) \subseteq C$ denote the set of selected centers whose SPU contains $q$ as an inner point and that were marked anomalous. We resolve overlap by averaging:
$$
\hat{q} \;=\;
\begin{cases}
q, & C(q) = \emptyset,\\[4pt]
\dfrac{1}{|C(q)|} \displaystyle\sum_{p \in C(q)} q^{(p)}, & C(q) \neq \emptyset,
\end{cases}
\tag{M3.5}
$$
where $q^{(p)}$ is the diffused image of $q$ inside the SPU centered at $p$. Points untouched by any anomalous SPU are guaranteed to be **exactly** preserved, which is why ASAP avoids the over-smoothing failure mode of whole-scene diffusion.

The output of M3 is
$$
\hat{P} \;=\; \big\{\hat{q} : q \in P\big\}.
$$

## 3.6 Pipeline algorithm

<!-- 给一个干净的伪代码 box，放进论文时可改成 algorithm2e 环境。 -->

The full ASAP pipeline is summarized below. Inputs in **bold** are the only learnable or calibrated quantities; all other quantities are computed analytically from $P$ at inference time.

```
Algorithm 1: ASAP inference
Input : LiDAR scan P; FPS stride eta; M1 hyperparameters {alpha, k, beta, r_min, r_max};
        M2 weights w = (w_c, w_a, w_v, w_d, b); M2 normalization (mu_*, sigma_*);
        ROC threshold tau*; M3 schedule beta(t), truncation t*; pretrained score s_theta.
Output: Purified point cloud P_hat.

1.  C <- FPS(P, stride=eta)                                # candidate SPU centers
2.  for each p in C in parallel:
3.      d_k(p) <- distance from p to its k-th nearest neighbor in P
4.      r_1(p) <- clip(alpha * d_k(p), [r_min, r_max])      # M1, eq. (M1.1)
5.      r_2(p) <- beta * r_1(p)                             # M1, eq. (M1.2)
6.      P_in(p), P_an(p) <- partition_SPU(P, p, r_1(p), r_2(p))
7.      f_c, f_a, f_v, f_d <- compute_features(P_in(p), P_an(p))   # M2.1-M2.4
8.      tildef_* <- (f_* - mu_*) / sigma_*                  # standardize
9.      s(p) <- sigmoid(w_c*tildef_c + w_a*tildef_a + w_v*tildef_v + w_d*tildef_d + b)  # M2.5
10. A <- {p in C : s(p) > tau*}                             # anomalous SPUs
11. for each p in A in parallel:
12.     X_p <- stack(P_in(p))
13.     tildeX_p <- (X_p - p) / r_2(p)
14.     hat_tildeX_p <- VP_SDE_purify(tildeX_p, t*, beta(.), s_theta)   # M3.1-M3.4
15.     hat_X_p <- r_2(p) * hat_tildeX_p + p
16. P_hat <- aggregate_overlap(P, A, {hat_X_p})              # M3.5
17. return P_hat
```

The detector is invoked separately as $\mathcal{D}(\hat{P})$, with no awareness of ASAP.

## 3.7 Complexity, memory, and detector compatibility

<!--
- KNN: O(N log N) with kd-tree, or O(N) on GPU via voxel hashing.
- 每个 SPU 4 个特征: O(m).
- 总 SPU 数 |C| ~ N / k (FPS 抽稀)。
- M3 扩散: 仅在 |A| 个 SPU 上跑 D 步。
- 检测器兼容性：输出仍是 R^3 点云。
-->

### 3.7.1 Computational complexity

Let $|C|$ denote the number of FPS-selected centers and $|A| \le |C|$ the number of anomalous SPUs. With $D$ reverse diffusion steps and an average inner size $\bar{m}$, the per-stage costs are:

- **KNN over $P$:** $O(N \log N)$ with a CPU kd-tree, or $O(N)$ amortized on GPU via hashed voxel grids.
- **Feature computation (M2):** $O(|C|\, \bar{m})$ for compactness, density, vMF concentration, and a $3 \times 3$ eigendecomposition per SPU.
- **Selective diffusion (M3):** $O(|A|\, D\, \bar{m})$, batched across anomalous SPUs.

Crucially, M3 cost scales with $|A|$ rather than with $N$. Whenever the attack is local (which is the regime of practical LiDAR attacks), $|A| \ll |C|$ and ASAP is significantly cheaper than whole-scene diffusion.

### 3.7.2 Memory footprint

At inference time, ASAP only needs (i) the input cloud $P$, (ii) the $k$-NN graph (sparse), and (iii) one batch of inner-point tensors of size $|A| \times \bar{m} \times 3$ for diffusion. No intermediate detector activation is touched.

### 3.7.3 Detector compatibility

Because $\hat{P}$ has the same $\mathbb{R}^3$ format as $P$, ASAP plugs in front of any LiDAR-only detector without retraining or architectural change. We instantiate $\mathcal{D}$ in the experiments (Sec. 4) with three families:

- **Pillar-based one-stage:** PointPillars.
- **Voxel-point two-stage:** PV-RCNN++.
- **Modern transformer / sparse voxel detectors:** DSVT and VoxelNeXt.

The fact that the same $(\alpha, k, \beta, r_{\min}, r_{\max}, \tau^{\star})$ work across all detectors is the empirical evidence behind the detector-agnostic claim in Section 4.
