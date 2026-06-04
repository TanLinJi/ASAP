# ASAP 项目状态分析与下一步建议

> 日期: 2026-05-31
> 目的: 全面评估当前实验进展、识别瓶颈、提出优化方向和文献需求

---

## 1. 当前实验进展总结

### 1.1 已完成 (可直接写入论文)

| 实验 | 状态 | 可用于论文 |
|------|------|-----------|
| E0 环境/数据 | KITTI + nuScenes 全部就绪 | 是 |
| E1 Clean baselines | 4 个检测器全部验证 (bit-exact) | 是 |
| E2 攻击生成 | KITTI 3 种攻击完成; nuScenes Injection + Perturbation(ε=0.5) 完成 | 是 |
| E3 SOR/ROR baselines | KITTI PointPillars + PV-RCNN 完成 | 是 |
| E4.1 KITTI/PointPillars | Track A policy 完整矩阵 | 是 |
| E4.2 KITTI/PV-RCNN | Track A revised policy + SOR/ROR | 是 |

### 1.2 进行中 / 待完成

| 实验 | 缺失内容 | 优先级 |
|------|----------|--------|
| **E4.3 nuScenes/VoxelNeXt ASAP** | no-defense 和 ROR 已有; ASAP 列 pending | **P0 — 论文主表必须** |
| **E4.4 nuScenes/TransFusion ASAP** | 同上 | **P0** |
| E5 消融实验 | Table 2 全部 [XX.X] | P1 — 4 页论文至少需要 1 个消融 |
| E6 推理延迟 | Table 3 全部 [XX.X] | P1 — reviewer 必问 |
| E3.3 Uniform SPU diffusion | 未跑 | P2 — 消融 A3 的对照组 |

### 1.3 关键数字汇总 (已验证)

**KITTI PointPillars (Car/Mod 3D AP_R40):**

| Attack | Clean | No-def | Best baseline | ASAP |
|--------|------:|-------:|--------------:|-----:|
| Injection | 78.40 | 60.10 | 66.07 (ROR) | **65.71** |
| Perturbation | 78.40 | 56.54 | 41.65 (ROR) | **61.04** |
| Dropping | 78.40 | 71.22 | 49.91 (ROR) | **71.22** (skip) |

**KITTI PV-RCNN (Car/Mod 3D AP_R40):**

| Attack | Clean | No-def | Best baseline | ASAP |
|--------|------:|-------:|--------------:|-----:|
| Injection | 84.36 | 63.16 | 72.77 (ROR) | **72.59** |
| Perturbation | 84.36 | 3.50 | 1.94 (ROR) | **10.96** |
| Dropping | 84.36 | 78.61 | 59.05 (ROR) | **78.61** (skip) |

---

## 2. 关键问题与风险

### 2.1 PV-RCNN Injection: ROR 略优于 ASAP

- ROR 72.77 vs ASAP 72.59 (差 0.18 AP)
- **影响**: 不能声称 ASAP 在所有行都是最优
- **建议**: 论文中改为声称 "cross-attack robustness" 而非 per-row dominance; 或者调优 injection filter 参数

### 2.2 PV-RCNN Perturbation: 绝对值仍然很低

- Clean 84.36 → No-def 3.50 → ASAP 10.96
- ASR 仍然 87%，恢复有限
- **影响**: 说明当前 geometric perturbation 攻击对 PV-RCNN 极其有效，ASAP 改善有限
- **建议**: 论文中诚实报告; 可以讨论为什么 voxel-point 两阶段架构对扰动更敏感

### 2.3 Dropping 策略: skip M3

- 当前 Dropping 行 = no-defense (不做任何净化)
- **影响**: reviewer 可能质疑 "为什么 ASAP 对 dropping 无效"
- **建议**: 论文中 frame 为 "conservative damage avoidance" — M2 检测到 dropping 模式后选择不干预，避免二次伤害。这是一个设计选择，不是失败。

### 2.4 nuScenes ASAP 列完全空白

- 这是论文 detector-agnostic claim 的核心支撑
- 没有 nuScenes ASAP 数字 = 无法声称跨数据集泛化
- **这是当前最大的 blocker**

### 2.5 Score-net v1 训练规模偏小

- 仅 16384 patches / 256 frames / 20 epochs
- 可能限制了泛化能力 (尤其是 nuScenes 的 32-beam 360° 布局)
- **问题**: 当前 score-net 是在 KITTI 上训练的，直接用于 nuScenes 是否合理？

---

## 3. 优化建议

### 3.1 实验优先级排序

```
P0 (必须完成才能投稿):
  1. nuScenes ASAP purification + eval (E4.3 + E4.4)
  2. 至少 1 个消融实验 (建议 A3: selective vs uniform)
  3. 推理延迟表 (E6)

P1 (强烈建议):
  4. M1 adaptive vs fixed radius 消融 (A1)
  5. 阈值 tau 敏感性 (A4)

P2 (如果时间允许):
  6. M2 单特征消融 (A2)
  7. Score-net v2 (更大训练集)
```

### 3.2 nuScenes ASAP 的技术决策

需要决定:
1. **Score-net 是否需要在 nuScenes 上重新训练?** 还是直接用 KITTI 训练的 checkpoint?
   - 如果跨数据集泛化，这本身就是一个 contribution point
   - 如果需要重训，需要额外 GPU 时间
2. **nuScenes 的 M2 阈值 tau 如何校准?** 需要 nuScenes 上的 calibration set
3. **Injection filter 参数是否需要调整?** nuScenes 点密度不同于 KITTI

### 3.3 论文写作优化

- **当前 references.bib**: 几乎所有条目标记 UNVERIFIED — 投稿前必须逐条验证
- **缺少关键引用**: 
  - PointPillars (Lang 2019) — 只在注释中提到，未正式加入 bib
  - PV-RCNN (Shi 2020)
  - VoxelNeXt (Chen 2023)
  - TransFusion (Bai 2022)
  - CenterPoint (Yin 2021) — 如果 related work 提到
  - DiffPure (Nie 2022) — 已有但需验证
  - Score-SDE (Song 2021) — 已有但需验证

---

## 4. 文献调研需求

### 4.1 必须下载并阅读的论文 (投稿前)

以下论文直接影响 novelty claim 和 related work 定位:

| # | 论文 | 为什么需要 | 优先级 |
|---|------|-----------|--------|
| 1 | **PointDP** (Sun et al., ICML 2023) | 最直接的竞争者 — 扩散净化 3D 点云 | 必须 |
| 2 | **LiDAR-SPD** (已有 PDF) | 球形投影 + 扩散防御，ASAP 的直接前驱 | 已有，需精读 |
| 3 | **Ada3Diff** (已有 PDF) | 自适应扩散净化，需要明确 ASAP 与其差异 | 已有，需精读 |
| 4 | **ScAR** (2023 preprint) | LiDAR 对抗鲁棒性 scaling，可能有重叠 claim | 高 |
| 5 | **3D-VField** (Lehner et al., CVPR 2022) | 点云对抗训练防御 | 中 |
| 6 | **IF-Defense** (Wu et al., CVPR 2021) | 隐式函数防御 3D 对抗攻击 | 高 |

### 4.2 需要确认 novelty 的关键对比点

ASAP 声称的三个 contribution 需要逐一确认没有被 scoop:

| Contribution | 潜在竞争者 | 需要确认 |
|---|---|---|
| 自适应半径 SPU | LiDAR-SPD 用固定球? Ada3Diff 用什么? | 精读两篇确认 |
| 无标签异常评分 + ROC 校准 | IF-Defense 用什么选择机制? | 需要读 IF-Defense |
| 选择性 VP-SDE (只净化异常区域) | PointDP 是全局还是局部? Ada3Diff 是自适应还是全局? | 精读确认 |

### 4.3 建议下载的论文清单

**高优先级 (直接竞争者/baseline):**
1. PointDP: https://arxiv.org/abs/2208.09801
2. IF-Defense: https://arxiv.org/abs/2010.05272
3. ScAR: https://arxiv.org/abs/2312.03085
4. 3D-VField: https://arxiv.org/abs/2112.04764

**中优先级 (方法论参考):**
5. DiffuBox (NeurIPS 2024): 点扩散用于检测精化
6. CloudFixer (ECCV 2024): 扩散引导的点云 TTA
7. Robust LiDAR survey (IJCV 2023): https://arxiv.org/abs/2212.10230

**低优先级 (broader context):**
8. Physically Realizable Adversarial Examples (Tu et al., CVPR 2020) — 已在 bib 中
9. DiffPure (Nie et al., ICML 2022) — 2D 图像扩散净化的开创性工作，已在 bib 中

---

## 5. 是否需要更详细的实验方案?

### 当前方案的充分性

`docs/03_icassp_paper_workflow.md` 已经是一个相当完整的执行计划。**不需要重写**，但以下方面可以补充:

### 5.1 建议补充: nuScenes ASAP 实验的具体执行方案

需要回答:
- Score-net 跨数据集策略 (直接迁移 vs 重训)
- nuScenes M2 校准流程
- 双卡并行策略 (purification + eval 的 GPU 分配)
- 预计 wall-clock 时间

### 5.2 建议补充: 消融实验的精简方案

4 页论文空间有限，建议只做:
- A3 (selective vs uniform) — 最能支撑 M2 gating 的价值
- A1 (adaptive vs fixed radius) — 如果只选一个固定半径做对比，选 r=0.20

### 5.3 不需要的

- 不需要重新设计攻击方案 (当前 geometric 攻击已足够，reviewer 可能建议 gradient-based 但那是 future work)
- 不需要额外的检测器 (4 个已经足够支撑 detector-agnostic claim)
- 不需要 Waymo (已正确 defer)

---

## 6. 建议的执行顺序

```
Week 1:
  Day 1-2: 决定 nuScenes score-net 策略 + 校准 M2
  Day 3-5: 跑 nuScenes ASAP purification (E4.3 VoxelNeXt + E4.4 TransFusion)
  Day 5:   跑 E3.3 Uniform SPU baseline (KITTI, 为消融准备)

Week 2:
  Day 1-2: 消融 A3 (selective vs uniform) + A1 (adaptive vs fixed)
  Day 2-3: 延迟测量 E6
  Day 3-5: 填充论文数字, 验证 references.bib

Week 3:
  Day 1-2: 论文修改 (method/experiments/abstract)
  Day 3:   /ars-reviewer 内部审稿
  Day 4-5: LaTeX 排版 + 最终 PDF
```

---

## 7. 总结

**当前最大 gap**: nuScenes ASAP 主表数字为空，这是投稿的 hard blocker。

**最值得优化的点**:
1. Score-net 跨数据集泛化策略 — 如果 KITTI-trained 直接用于 nuScenes 且有效，这本身就是 novelty
2. Dropping 的 framing — 需要在论文中清晰解释为什么 skip 是正确的设计选择
3. PV-RCNN Perturbation 的低恢复率 — 需要诚实讨论

**文献方面**: 最紧迫的是下载并精读 PointDP 和 IF-Defense，确认 ASAP 的 selective purification 机制确实是新的。
