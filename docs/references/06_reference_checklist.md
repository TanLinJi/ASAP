# ASAP 论文参考文献补充清单

> 日期: 2026-05-31
> 目的: 列出 ICASSP 投稿所需的全部参考文献，标注已有/待下载状态

---

## 当前 references.bib 中已有的条目

| BibKey | 论文 | 状态 |
|--------|------|------|
| banerjee2005clustering | vMF Clustering (JMLR 2005) | ✅ 已验证 |
| youden1950index | Youden Index (Cancer 1950) | ✅ 已验证 |
| song2021scorebased | Score-Based SDE (ICLR 2021) | ✅ 需验证细节 |
| nie2022diffpure | DiffPure (ICML 2022) | ✅ 需验证细节 |
| geiger2012kitti | KITTI (CVPR 2012) | ⚠️ UNVERIFIED |
| sun2020waymo | Waymo Open Dataset (CVPR 2020) | ⚠️ UNVERIFIED |
| caesar2020nuscenes | nuScenes (CVPR 2020) | ⚠️ UNVERIFIED |
| tu2020physically | Physically Realizable Attack (CVPR 2020) | ⚠️ UNVERIFIED |
| cao2019adversarial | LiDAR Spoofing Attack (CCS 2019) | ⚠️ UNVERIFIED |

---

## 需要补充的参考文献 (按论文章节分类)

### A. 3D 检测器 (Introduction + Experiments)

论文中使用了 4 个检测器，每个都需要引用。

| # | 论文 | 会议/期刊 | 年份 | 需要下载? |
|---|------|----------|------|----------|
| 1 | **PointPillars: Fast Encoders for Object Detection from Point Clouds** — Lang et al. | CVPR | 2019 | 需要 |
| 2 | **PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection** — Shi et al. | CVPR | 2020 | 需要 |
| 3 | **VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking** — Chen et al. | CVPR | 2023 | 需要 |
| 4 | **TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers** — Bai et al. | CVPR | 2022 | 需要 |
| 5 | **Center-based 3D Object Detection and Tracking (CenterPoint)** — Yin et al. | CVPR | 2021 | 需要 (related work 中提到) |

arXiv/下载链接:
- PointPillars: https://arxiv.org/abs/1812.05784
- PV-RCNN: https://arxiv.org/abs/1912.13192
- VoxelNeXt: https://arxiv.org/abs/2303.11301
- TransFusion: https://arxiv.org/abs/2203.11496
- CenterPoint: https://arxiv.org/abs/2006.11275

### B. LiDAR 对抗攻击 (Introduction §1.1 + Related Work §2.1)

| # | 论文 | 会议/期刊 | 年份 | 需要下载? |
|---|------|----------|------|----------|
| 6 | **Adversarial Objects Against LiDAR-Based Autonomous Driving Systems (LiDAR Spoofing)** — Cao et al. | CCS | 2019 | 需要 (已在 bib 但 UNVERIFIED) |
| 7 | **Physically Realizable Adversarial Examples for LiDAR Object Detection** — Tu et al. | CVPR | 2020 | 需要 (已在 bib 但 UNVERIFIED) |
| 8 | **Towards Robust LiDAR-based Perception in Autonomous Driving: General Black-box Adversarial Sensor Attack and Countermeasures** — Sun et al. | USENIX Security | 2020 | 需要 |
| 9 | **Adversarial Point Cloud Perturbations against 3D Object Detection** — Wang et al. | Neurocomputing | 2021 | 需要 |
| 10 | **Generating 3D Adversarial Point Clouds** — Xiang et al. | CVPR | 2019 | 需要 |
| 11 | **Adversarial Obstacle Generation Against LiDAR-Based 3D Object Detection** — Wang et al. | IEEE Trans. Multimedia | 2024 | 可选 (recent) |
| 12 | **You Can't See Me: Physical Removal Attacks on LiDAR-based AV Frameworks** — Cao et al. | USENIX Security | 2023 | 可选 (dropping 相关) |

arXiv/下载链接:
- Cao 2019: https://arxiv.org/abs/1907.06826 (或 CCS proceedings)
- Tu 2020: https://arxiv.org/abs/2004.00543
- Sun 2020: https://www.usenix.org/conference/usenixsecurity20/presentation/sun
- Wang 2021: https://doi.org/10.1016/j.neucom.2020.12.114
- Xiang 2019: https://arxiv.org/abs/1809.07016
- Wang 2024: IEEE Trans. Multimedia (DOI pending)
- Cao 2023: https://www.usenix.org/conference/usenixsecurity23/presentation/cao

### C. 点云防御方法 (Related Work §2.2)

| # | 论文 | 会议/期刊 | 年份 | 需要下载? |
|---|------|----------|------|----------|
| 13 | **IF-Defense: 3D Adversarial Point Cloud Defense via Implicit Function based Restoration** — Wu et al. | CVPR | 2021 | ✅ 已有 PDF |
| 14 | **3D-VField: Adversarial Augmentation of Point Clouds for Domain Generalization in 3D Object Detection** — Lehner et al. | CVPR | 2022 | 需要 |
| 15 | **DUP-Net: Denoiser and Upsampler Network for 3D Adversarial Point Clouds Defense** — Zhou et al. | ICCV | 2019 | 需要 |
| 16 | **LiDAR-SPD: Improving Adversarial Robustness of 3D Object Detection via Spherical Projection and Diffusion** — Cai et al. | ICASSP | 2025 | ✅ 已有 PDF |

arXiv/下载链接:
- IF-Defense: 已有
- 3D-VField: https://arxiv.org/abs/2112.04764
- DUP-Net: ICCV 2019 proceedings (搜索 "DUP-Net Zhou 2019 ICCV")
- LiDAR-SPD: 已有

### D. 扩散模型与点云净化 (Related Work §2.3 + Method §3.5)

| # | 论文 | 会议/期刊 | 年份 | 需要下载? |
|---|------|----------|------|----------|
| 17 | **PointDP: Diffusion-driven Purification against Adversarial Attacks on 3D Point Cloud Recognition** — Sun et al. | ICML | 2023 | ✅ 已有 PDF |
| 18 | **Ada3Diff: Defending against 3D Adversarial Point Clouds via Adaptive Diffusion** — Zhang et al. | ACM MM | 2023 | ✅ 已有 PDF |
| 19 | **Diffusion Models-Based Purification for Common Corruptions on Robust 3D Object Detection (LiDARPure)** — Cai et al. | Sensors | 2024 | ✅ 已有 PDF |
| 20 | **Diffusion Probabilistic Models for 3D Point Cloud Generation** — Luo & Hu | CVPR | 2021 | 需要 |
| 21 | **DiffPure: Diffusion Models for Adversarial Purification** — Nie et al. | ICML | 2022 | 需要 (已在 bib 但需验证) |
| 22 | **Point Cloud Layerwise Diffusion for Adversarial Purification (PCLD)** — 2024 preprint | arXiv | 2024 | 可选 |
| 23 | **Denoising Diffusion Probabilistic Models (DDPM)** — Ho et al. | NeurIPS | 2020 | 需要 (扩散基础引用) |

arXiv/下载链接:
- PointDP: https://arxiv.org/abs/2208.09801 (已有)
- Ada3Diff: 已有
- LiDARPure: 已有
- Luo 2021: https://arxiv.org/abs/2103.01458
- DiffPure: https://arxiv.org/abs/2205.07460
- PCLD: https://arxiv.org/abs/2403.06698
- DDPM: https://arxiv.org/abs/2006.11239

### E. 数据集 (Experiments §4.1)

| # | 论文 | 会议/期刊 | 年份 | 需要下载? |
|---|------|----------|------|----------|
| 24 | **Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite** — Geiger et al. | CVPR | 2012 | 需要 (已在 bib 但 UNVERIFIED) |
| 25 | **nuScenes: A Multimodal Dataset for Autonomous Driving** — Caesar et al. | CVPR | 2020 | 需要 (已在 bib 但 UNVERIFIED) |

arXiv/下载链接:
- KITTI: https://www.cvlibs.net/publications/Geiger2012CVPR.pdf
- nuScenes: https://arxiv.org/abs/1903.11027

### F. 工具与实现 (Experiments §4.1)

| # | 论文 | 类型 | 年份 | 需要下载? |
|---|------|------|------|----------|
| 26 | **OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds** — Team OpenPCDet | GitHub/Tech Report | 2020 | 不需要下载，引用 GitHub |

引用格式:
```bibtex
@misc{openpcdet2020,
  title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
  author={OpenPCDet Development Team},
  howpublished={\url{https://github.com/open-mmlab/OpenPCDet}},
  year={2020}
}
```

### G. 方法论基础 (Method §3.3-3.5)

| # | 论文 | 会议/期刊 | 年份 | 需要下载? |
|---|------|----------|------|----------|
| 27 | **Score-Based Generative Modeling through Stochastic Differential Equations** — Song et al. | ICLR | 2021 | 需要 (已在 bib 但需验证) |
| 28 | **Clustering on the Unit Hypersphere using von Mises-Fisher Distributions** — Banerjee et al. | JMLR | 2005 | ✅ 已在 bib |
| 29 | **Index for Rating Diagnostic Tests (Youden-J)** — Youden | Cancer | 1950 | ✅ 已在 bib |
| 30 | **Towards 3D Point Cloud Based Object Maps for Household Environments (PCL/SOR)** — Rusu et al. | Robotics and Autonomous Systems | 2008 | 需要 |

arXiv/下载链接:
- Song 2021: https://arxiv.org/abs/2011.13456
- Rusu 2008: 搜索 "Rusu 2008 Towards 3D Point Cloud Based Object Maps" (Springer)

---

## 总结: 需要你下载的论文

### 必须下载 (论文中直接引用)

| 优先级 | 论文 | 最简获取方式 |
|--------|------|-------------|
| **P0** | PointPillars (Lang 2019) | https://arxiv.org/abs/1812.05784 |
| **P0** | PV-RCNN (Shi 2020) | https://arxiv.org/abs/1912.13192 |
| **P0** | VoxelNeXt (Chen 2023) | https://arxiv.org/abs/2303.11301 |
| **P0** | TransFusion (Bai 2022) | https://arxiv.org/abs/2203.11496 |
| **P0** | Sun 2020 USENIX attack | https://www.usenix.org/conference/usenixsecurity20/presentation/sun |
| **P0** | 3D-VField (Lehner 2022) | https://arxiv.org/abs/2112.04764 |
| **P0** | Luo 2021 点云扩散 | https://arxiv.org/abs/2103.01458 |
| **P0** | DDPM (Ho 2020) | https://arxiv.org/abs/2006.11239 |

### 建议下载 (增强 related work 完整性)

| 优先级 | 论文 | 最简获取方式 |
|--------|------|-------------|
| **P1** | CenterPoint (Yin 2021) | https://arxiv.org/abs/2006.11275 |
| **P1** | DUP-Net (Zhou 2019) | ICCV 2019 proceedings |
| **P1** | Wang 2021 perturbation attack | Neurocomputing DOI |
| **P1** | Xiang 2019 3D adversarial | https://arxiv.org/abs/1809.07016 |
| **P1** | DiffPure (Nie 2022) | https://arxiv.org/abs/2205.07460 |

### 可选 (如果空间允许)

| 优先级 | 论文 | 最简获取方式 |
|--------|------|-------------|
| P2 | PCLD (2024 preprint) | https://arxiv.org/abs/2403.06698 |
| P2 | Cao 2023 removal attack | USENIX Security 2023 |
| P2 | Wang 2024 obstacle generation | IEEE Trans. Multimedia |
| P2 | Survey: Physical Adversarial Attacks on LiDAR (2024) | https://arxiv.org/abs/2409.20426 |

---

## 预计最终引用数量

ICASSP 4 页论文通常有 **20-30 条引用**。按上述清单：

- 已有 (bib 中): 9 条
- P0 必须补充: 8 条
- P1 建议补充: 5 条
- **总计: 22-27 条** — 符合 ICASSP 标准

---

## 下载后的存放位置

请将 PDF 放入 `/root/autodl-tmp/ASAP/docs/references/`，文件名建议格式:
```
Author_Year_ShortTitle.pdf
```
例如:
- `Lang_2019_PointPillars.pdf`
- `Shi_2020_PV-RCNN.pdf`
- `Chen_2023_VoxelNeXt.pdf`

下载完成后告诉我，我会帮你验证 BibTeX 条目并更新 `references.bib`。
