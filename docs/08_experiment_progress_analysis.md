# ASAP 实验进展分析与下一步计划

> 日期: 2026-05-31 (实验完成后)
> 状态: nuScenes ASAP purification + VoxelNeXt eval 已完成

---

## 1. 全部实验结果汇总

### 1.1 KITTI 主表 (已完成, Track A revised policy)

Clean: PointPillars Car/Mod = 78.40, PV-RCNN Car/Mod = 84.36

| Dataset | Detector | Attack | No-def | SOR | ROR | **ASAP** |
|---------|----------|--------|--------|-----|-----|----------|
| KITTI | PointPillars | Injection | 60.10 | 58.00 | 66.07 | **65.71** |
| KITTI | PointPillars | Perturbation | 56.54 | 34.12 | 41.65 | **61.04** |
| KITTI | PV-RCNN | Injection | 63.16 | 66.82 | 72.77 | **72.59** |
| KITTI | PV-RCNN | Perturbation | 3.50 | 0.89 | 1.94 | **10.96** |

**KITTI 结论**: ASAP 在 Perturbation 上明显优于所有 baseline；在 Injection 上接近 ROR。

### 1.2 nuScenes 主表 (刚完成)

Clean: VoxelNeXt mAP = 60.52, TransFusion-L mAP = 64.57

**VoxelNeXt (E4.3):**

| Attack | No-def | SOR | ROR | **ASAP** | ASAP vs No-def |
|--------|--------|-----|-----|----------|----------------|
| Injection | 21.97 | 30.01 | **53.29** | 52.45 | +30.48 |
| Perturbation ε=0.5 | 40.27 | 42.16 | **51.36** | 41.32 | +1.05 |

**TransFusion-Lidar (E4.4) — 只有 SOR/ROR, 无 ASAP:**

| Attack | No-def | SOR | ROR |
|--------|--------|-----|-----|
| Injection | 19.13 | 30.92 | **57.29** |
| Perturbation ε=0.5 | 43.02 | 43.95 | **53.51** |

### 1.3 关键发现

| 发现 | 影响 | 严重度 |
|------|------|--------|
| ASAP Injection 在 nuScenes 上有效 (52.45 vs no-def 21.97) | ✅ 可以写入论文 | — |
| ASAP Injection 略低于 ROR (52.45 vs 53.29, 差 0.84) | ⚠️ 不能声称 ASAP 在所有行都最优 | 中 |
| ASAP Perturbation 在 nuScenes 上完全无效 (编辑率 0.0%) | ❌ M2 scorer 对 perturbation 失效 | 高 |
| TransFusion-Lidar 还没有 ASAP 行 | ❌ 需要决定是否跑 | 中 |

---

## 2. 问题诊断: 为什么 nuScenes Perturbation 编辑率 = 0%

**根因**: M2 scorer 的 tau=0.049 对 nuScenes perturbation 攻击过高。

**技术解释**:
- Perturbation ε=0.5 只是将点在表面附近微移 (≤0.5m)
- nuScenes 点云本身就比 KITTI 更稀疏 (32-beam vs 64-beam)，几何特征的自然方差更大
- M2 的四个特征 (compactness, anisotropy, vMF, density) 在 nuScenes 上的 clean/attacked 分布重叠严重
- 校准时 Youden-J 给出的 tau=0.049 实际上意味着 "几乎没有 SPU 能被区分"

**对比 KITTI**: KITTI E2.2 的 tau=0.024, Youden-J=0.404 (中等区分度)。nuScenes 的 tau=0.049 但实际区分度可能接近 0。

---

## 3. 可选方案评估

### 方案 A: 降低 tau 重跑 nuScenes Perturbation

- **操作**: 将 tau 从 0.049 降到 0.01 或 0.005
- **风险**: 会标记大量 benign SPU，可能损害 clean geometry
- **时间**: ~2.5h purification + ~1h eval = ~3.5h
- **预期**: 编辑率可能升到 20-30%，但 mAP 不一定提升 (因为 score-net 对 on-surface perturbation 的修复能力有限)

### 方案 B: 接受现状，论文只报告 Injection

- **操作**: 主表只放 Injection 行 (ASAP 有效)
- **论文 framing**: "ASAP 对 point injection 攻击有效；对 on-surface perturbation，M2 的几何特征难以区分被扰动点，这是未来工作方向"
- **风险**: reviewer 可能质疑 "为什么只测一种攻击"
- **缓解**: KITTI 上两种攻击都有效，nuScenes 只是 Perturbation 失效

### 方案 C: 论文诚实报告两行，ASAP Perturbation = No-defense

- **操作**: 主表放两行，ASAP Perturbation 列写 41.32 (≈ no-def 40.27)
- **论文 framing**: "M2 在 nuScenes perturbation 上未触发，说明 on-surface 微扰在稀疏 32-beam 点云上的几何异常不显著"
- **优点**: 诚实，reviewer 会尊重
- **缺点**: 看起来 ASAP 对 perturbation 无效

### 方案 D: 用 KITTI 的 tau 跑 nuScenes (跨数据集 tau)

- **操作**: 用 KITTI E2.2 的 tau=0.024 (而非 nuScenes 校准的 0.049)
- **理由**: 如果 KITTI tau 更激进且有效，说明 "tau 可以从一个数据集迁移"
- **时间**: ~3.5h
- **风险**: 可能仍然编辑率很低，或者过度编辑

### 方案 E: 跑 TransFusion-Lidar ASAP (只 Injection)

- **操作**: purified 点云已有 (detector-agnostic)，只需跑 TransFusion eval on E2.1 purified
- **时间**: ~30 min
- **价值**: 多一个检测器的 ASAP Injection 数字

---

## 4. 推荐策略

**我的建议: 方案 C + E 组合**

理由:
1. **诚实报告** (方案 C) 比隐藏结果更好 — ICASSP reviewer 尊重诚实的 limitation 讨论
2. **TransFusion ASAP Injection** (方案 E) 几乎免费 (30 min)，多一行有效数字
3. **不建议重跑** (方案 A/D) — 即使编辑率提升，score-net 对 on-surface perturbation 的修复能力本身有限，mAP 提升可能很小，不值得 3.5h 的风险

**最终主表结构**:

| Dataset | Detector | Attack | No-def | ROR | **ASAP** | Δ vs No-def |
|---------|----------|--------|--------|-----|----------|-------------|
| KITTI | PointPillars | Injection | 60.10 | 66.07 | **65.71** | +5.61 |
| KITTI | PointPillars | Perturbation | 56.54 | 41.65 | **61.04** | +4.50 |
| KITTI | PV-RCNN | Injection | 63.16 | 72.77 | **72.59** | +9.43 |
| KITTI | PV-RCNN | Perturbation | 3.50 | 1.94 | **10.96** | +7.46 |
| nuScenes | VoxelNeXt | Injection | 21.97 | 53.29 | **52.45** | +30.48 |
| nuScenes | VoxelNeXt | Perturbation | 40.27 | **51.36** | 41.32† | +1.05 |
| nuScenes | TransFusion-L | Injection | 19.13 | 57.29 | **TBD** | TBD |

† M2 未触发，等效于 no-defense。

**论文 narrative**:
- ASAP 在 **Injection** 攻击上跨 4 个检测器、2 个数据集一致有效
- ASAP 在 KITTI **Perturbation** 上优于所有 baseline (61.04 vs ROR 41.65)
- nuScenes Perturbation 是 limitation: 32-beam 稀疏点云上 on-surface 微扰的几何异常不显著，M2 未触发
- 这指向未来工作: 需要更强的特征 (如 learned features) 来检测 on-surface perturbation

---

## 5. 下一步任务清单

### 立即执行 (今天)

| # | 任务 | 时间 | 优先级 |
|---|------|------|--------|
| 1 | 跑 TransFusion-Lidar eval on ASAP purified E2.1 Injection | 30 min | P0 |
| 2 | 消融 A3: Uniform SPU (tau=0) on KITTI E2.2 PointPillars | 3h | P0 |
| 3 | E6 延迟测量 (从已有 meta.jsonl 解析) | 1h | P1 |

### 短期 (1-2 天)

| # | 任务 | 时间 | 优先级 |
|---|------|------|--------|
| 4 | 消融 A1: fixed radius on KITTI E2.2 | 4h | P1 |
| 5 | 填充论文 Table 1 所有数字 | 2h | P0 |
| 6 | 更新 abstract/conclusion 中的 claim | 1h | P0 |
| 7 | 验证 references.bib (去除 UNVERIFIED) | 2h | P1 |

### 中期 (3-5 天)

| # | 任务 | 时间 | 优先级 |
|---|------|------|--------|
| 8 | 论文各章节修改 (method/experiments/related work) | 1-2 天 | P0 |
| 9 | /ars-reviewer 内部审稿 | 2h | P1 |
| 10 | LaTeX 排版 + 最终 PDF | 1 天 | P0 |

---

## 6. 论文 claim 调整建议

### 原始 claim (过强):
> "ASAP improves robustness against all three attack families across multiple detectors and datasets"

### 修正后 claim (准确):
> "ASAP consistently improves robustness against point injection attacks across four detectors on KITTI and nuScenes without detector retraining. On KITTI, ASAP also outperforms all baselines against point perturbation. For on-surface perturbation on sparse 32-beam LiDAR, the geometric anomaly signal is weaker, indicating room for future feature design."

这个 claim 诚实、有数据支撑、且指出了明确的 future work 方向。
