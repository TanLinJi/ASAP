# 3. ASAP Framework

## 3.1 Overview

Let $P=\{p_i\}_{i=1}^{N}\subset\mathbb{R}^3$ be an input LiDAR scan and let $\mathcal{D}$ be an off-the-shelf detector. ASAP returns a purified point cloud $\hat{P}$ that is passed to $\mathcal{D}$ with no detector retraining or architectural change. The framework has three modules.

- **M1: Adaptive Radius.** Build local Spherical Purification Units (SPUs) whose radius follows local point density.
- **M2: Anomaly-Selective Structure Gate.** Score every SPU with lightweight geometric statistics and edit only high-risk units.
- **M3: Selective Diffusion.** Apply a short VP-SDE score-net purifier inside selected SPUs, with an optional support filter for sparse off-surface artifacts.

The main design principle is structure preservation: clean or low-risk local geometry is copied exactly, while suspicious local regions are denoised before the detector sees the scan.

## 3.2 Adaptive Radius (M1)

LiDAR density changes strongly with range, so a single global radius can either under-populate sparse regions or over-edit dense surfaces. For an SPU center $p$, let $p_{(k)}(p)$ be the $k$-th nearest neighbor in $P\setminus\{p\}$ and define
$$
d_k(p)=\lVert p-p_{(k)}(p)\rVert_2 .
$$
Under a locally Poisson sampling model, $d_k(p)\propto \rho(p)^{-1/3}$ for local point density $\rho(p)$, so it is a natural local scale estimate. ASAP sets the outer and inner SPU radii as
$$
r_1(p)=\mathrm{clip}\!\left(\alpha d_k(p),[r_{\min},r_{\max}]\right),\qquad
r_2(p)=\beta r_1(p),
\tag{1}
$$
where $\alpha>1$ and $\beta\in(0,1)$. The clipping prevents degenerate tiny SPUs in dense regions and overly large units in sparse regions. Candidate centers $C\subset P$ are selected by farthest point sampling with stride $\eta$.

Each SPU is split into an editable inner ball and a context annulus:
$$
\mathcal{S}(p)=\{q\in P:\lVert q-p\rVert_2\le r_1(p)\},\quad
P_{\mathrm{in}}(p)=\{q\in P:\lVert q-p\rVert_2\le r_2(p)\},
\tag{2}
$$
with $P_{\mathrm{an}}(p)=\mathcal{S}(p)\setminus P_{\mathrm{in}}(p)$. Only $P_{\mathrm{in}}(p)$ can be edited by M3; annulus points stabilize M2 statistics but are not directly diffused. We use $\alpha=2$, $k=16$, $\beta=0.67$, $r_{\min}=0.10$m, and $r_{\max}=0.40$m unless stated otherwise.

## 3.3 Anomaly-Selective Structure Gate (M2)

M2 prevents local diffusion from becoming whole-scene smoothing. For each SPU, it computes four analytic, detector-agnostic features:
$$
f_c(p)=\frac{|P_{\mathrm{in}}(p)|}{V_2(p)},
\tag{3}
$$
where $V_2(p)=\frac{4}{3}\pi r_2(p)^3$ is the inner volume;
$$
f_a(p)=1-\frac{\lambda_3}{\lambda_1+\lambda_2+\lambda_3},
\tag{4}
$$
where $\lambda_1\ge\lambda_2\ge\lambda_3$ are eigenvalues of the covariance of $P_{\mathrm{in}}(p)$;
$$
f_v(p)=\hat{\kappa}(p)=\frac{\bar{R}(3-\bar{R}^2)}{1-\bar{R}^2},
\tag{5}
$$
where $\bar{R}$ is the mean resultant length of unit directions $(q-p)/\lVert q-p\rVert_2$ and $\hat{\kappa}$ is the standard vMF concentration estimate [@banerjee2005clustering]; and
$$
f_d(p)=
\frac{|P_{\mathrm{in}}(p)|/V_2(p)}
{|P_{\mathrm{an}}(p)|/V_{\mathrm{an}}(p)+\varepsilon},
\tag{6}
$$
where $V_{\mathrm{an}}(p)=\frac{4}{3}\pi(r_1(p)^3-r_2(p)^3)$. These features capture compact clusters, low-rank local structures, angular concentration, and inner/outer density discontinuities.

Raw features are standardized using clean-reference statistics,
$$
\tilde{f}_\ast(p)=\frac{f_\ast(p)-\mu_\ast}{\sigma_\ast},\qquad \ast\in\{c,a,v,d\},
$$
and aggregated by a five-parameter logistic scorer:
$$
s(p)=\sigma\!\left(w_c\tilde{f}_c+w_a\tilde{f}_a+w_v\tilde{f}_v+w_d\tilde{f}_d+b\right).
\tag{7}
$$
Only this small aggregator and its decision threshold are calibrated; no detector parameter or activation is used. Given held-out SPU labels, the operating point is chosen by the Youden index [@youden1950index],
$$
\tau^\star=\arg\max_\tau\{\mathrm{TPR}(\tau)-\mathrm{FPR}(\tau)\}.
\tag{8}
$$
SPUs with $s(p)\le\tau^\star$ are copied exactly. Section 4 reports that M2 should be interpreted as a cross-attack analytic gate: density ratio dominates the KITTI perturbation cue, but nonselective diffusion collapses without M2 selection.

## 3.4 Selective Diffusion (M3)

For a selected SPU, let $X_p\in\mathbb{R}^{m\times3}$ stack the coordinates of $P_{\mathrm{in}}(p)$. We normalize it into the unit ball,
$$
\tilde{X}_p=\frac{X_p-p}{r_2(p)}.
\tag{9}
$$
M3 uses a Variance-Preserving SDE [@song2021scorebased],
$$
\mathrm{d}x_t=-\frac{1}{2}\beta(t)x_t\,\mathrm{d}t+\sqrt{\beta(t)}\,\mathrm{d}w_t,
\tag{10}
$$
whose marginal satisfies
$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I).
\tag{11}
$$
Following adversarial purification with short diffusion trajectories [@nie2022diffpure], ASAP forwards each normalized local patch only to a truncation time $t^\star$ and then applies the learned reverse score $s_\theta(x_t,t)$ to obtain $\hat{\tilde{X}}_p$. The result is mapped back to world coordinates:
$$
\hat{X}_p=r_2(p)\hat{\tilde{X}}_p+p.
\tag{12}
$$

Because SPUs overlap, a point may receive multiple purified proposals. Let $C(q)$ be the selected centers whose inner SPU contains $q$, and let $q^{(p)}$ be the proposal from center $p$. ASAP aggregates by
$$
\hat{q}=
\begin{cases}
q, & C(q)=\emptyset,\\
\frac{1}{|C(q)|}\sum_{p\in C(q)}q^{(p)}, & C(q)\ne\emptyset.
\end{cases}
\tag{13}
$$
This guarantees exact preservation for points outside anomalous SPUs.

For attacks that leave sparse unsupported artifacts, ASAP can append a support filter:
$$
\hat{q}\in\hat{P}_{\mathrm{out}}
\Longleftrightarrow
\left|\{\hat{q}'\ne\hat{q}:\lVert\hat{q}'-\hat{q}\rVert_2\le r_f\}\right|\ge m_f .
\tag{14}
$$
When enabled in Section 4, this branch is named **ASAP-pfilter** and reported as a score-net + support-filter cascade using the same $(r_f,m_f)$ as the ROR baseline. It is not presented as pure diffusion beating ROR.

## 3.5 Inference Algorithm

```
Algorithm 1: ASAP inference
Input : LiDAR scan P; M1 parameters; M2 scorer and threshold tau*;
        score network s_theta; optional support-filter parameters.
Output: Purified point cloud P_hat.

1.  C <- FPS(P, stride=eta)
2.  for p in C:
3.      compute r1(p), r2(p) by Eq. (1)
4.      split S(p) into P_in(p) and P_an(p)
5.      compute f_c, f_a, f_v, f_d by Eqs. (3)-(6)
6.      compute s(p) by Eq. (7)
7.  A <- {p in C : s(p) > tau*}
8.  for p in A:
9.      normalize P_in(p), run short VP-SDE purification, map back
10. aggregate overlapping proposals by Eq. (13)
11. if the selected policy enables support filtering:
12.     apply Eq. (14)
13. return P_hat
```

The detector is invoked only after purification as $\mathcal{D}(\hat{P})$. Since $\hat{P}$ is still a raw point cloud, ASAP is compatible with pillar, voxel, voxel-point, and transformer-query LiDAR detectors. Its computational bottleneck is M3, which scales with the number of anomalous SPUs rather than with the full scan whenever $|A|\ll|C|$.
