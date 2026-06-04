# ASAP ICASSP 投稿实验策略

> 日期: 2026-05-31
> 目的: 从广度、难度、深度三个维度制定确保 ICASSP 投稿顺利的实验路线

---

## 0. 竞争者分析 (基于精读参考文献)

精读 6 篇核心参考文献后，ASAP 的竞争定位如下：

| 方法 | 任务 | 粒度 | 扩散方式 | 选择机制 | 数据集 | 检测器 |
|------|------|------|---------|---------|-------|--------|
| **PointDP** (ICLR 2023) | 分类 | 全局整体 | 条件扩散 (encoder+latent) | 无 (全点云) | ModelNet40 | 无 (分类器) |
| **IF-Defense** (CVPR 2021) | 分类 | 全局优化 | 无扩散 (隐式函数+分布loss) | SOR 预处理 | ModelNet40 | 无 (分类器) |
| **Ada3Diff** (ACM MM 2023) | 分类 | 全局整体 | 自适应时间步扩散 | 扰动估计选时间步 | ModelNet40, ShapePart | 无 (分类器) |
| **LiDAR-SPD** (ICASSP 2025) | 检测 | SPU 局部 | VP-SDE 逐球扩散 | 球形投影遮挡检测 | KITTI | PP/PV-RCNN/CP |
| **LiDARPure** (Sensors 2024) | 检测 | 体素全局 | 体素级扩散 | 无 (全场景) | KITTI | PP/PV-RCNN/CP |
| **ASAP (ours)** | 检测 | SPU 局部 | VP-SDE 选择性扩散 | 异常评分+ROC校准 | KITTI+nuScenes | 4 检测器 |

### ASAP 相对于每个竞争者的差异化

1. **vs PointDP**: PointDP 做分类，全局扩散，无选择机制。ASAP 做检测，局部选择性扩散，有 M2 异常评分门控。**完全不同的任务和粒度。**

2. **vs IF-Defense**: IF-Defense 用隐式函数优化坐标，需要 200 次迭代优化，只做分类。ASAP 是前向推理一次，做检测。**不构成直接竞争。**

3. **vs Ada3Diff**: Ada3Diff 的自适应是选择扩散时间步 (全点云都扩散，只是步数不同)。ASAP 的自适应是选择哪些区域扩散 (大部分区域不动)。**核心机制不同。**

4. **vs LiDAR-SPD** (最直接竞争者, ICASSP 2025):
   - LiDAR-SPD 用**固定半径** r₁=0.15m 构建 SPU → ASAP 用**自适应半径**
   - LiDAR-SPD 用**球形投影遮挡**检测注入点 → ASAP 用**四特征异常评分** (更通用，覆盖 perturbation/dropping)
   - LiDAR-SPD **所有 SPU 都扩散** → ASAP **只扩散异常 SPU** (选择性)
   - LiDAR-SPD 只在 KITTI 上验证 → ASAP 在 KITTI + nuScenes 上验证
   - LiDAR-SPD 只对 injection 和 perturbation 有效 → ASAP 对 dropping 有保守策略
   - **结论: ASAP 是 LiDAR-SPD 的明确改进，novelty 安全**

5. **vs LiDARPure**: LiDARPure 做体素级全场景扩散，针对 common corruptions (雾/雪/噪声)，不是对抗攻击。**不同问题设定。**

### Novelty 结论

ASAP 的三个 contribution 均安全：
- ✅ 自适应半径 SPU (LiDAR-SPD 用固定半径)
- ✅ 无标签异常评分 + ROC 校准选择 (LiDAR-SPD 用遮挡投影，只能检测 injection)
- ✅ 选择性扩散 (所有竞争者都是全局/全 SPU 扩散)

---

## 1. ICASSP 审稿标准与实验要求

ICASSP 是信号处理顶会，4 页正文。审稿关注：

1. **Technical novelty** — 方法是否有新意 (已确认安全)
2. **Experimental validation** — 实验是否充分支撑 claim
3. **Reproducibility** — 实验设置是否清晰可复现
4. **Presentation** — 4 页内是否讲清楚

对于 4 页论文，实验部分通常 1-1.5 栏，需要：
- 1 张主结果表 (必须)
- 1 张消融表或延迟表 (强烈建议)
- 可选: 1 张可视化图

---

## 2. 实验广度: 最小充分矩阵

### 2.1 当前 claim 需要的最小实验支撑

论文声称 "detector-agnostic" + "cross-dataset"，最小矩阵：

| 维度 | 最小要求 | 当前状态 |
|------|---------|---------|
| 数据集 | ≥2 个 | KITTI ✅ + nuScenes (ASAP pending) |
| 检测器 | ≥3 个 | PointPillars ✅ + PV-RCNN ✅ + VoxelNeXt/TransFusion (pending) |
| 攻击类型 | ≥2 个 | Injection ✅ + Perturbation ✅ + Dropping (skip策略) |
| Baseline | ≥2 个 | SOR ✅ + ROR ✅ |

### 2.2 推荐的最终主表结构

考虑到 4 页空间限制，建议主表精简为：

```
Table 1: Main results. Cell = mAP / ASR(%).
                    
Dataset  | Detector      | Attack  | No-def | ROR   | ASAP
---------|---------------|---------|--------|-------|------
KITTI    | PointPillars  | Inj     | ✅     | ✅    | ✅
KITTI    | PointPillars  | Pert    | ✅     | ✅    | ✅
KITTI    | PV-RCNN       | Inj     | ✅     | ✅    | ✅
KITTI    | PV-RCNN       | Pert    | ✅     | ✅    | ✅
nuScenes | VoxelNeXt     | Inj     | ✅     | ✅    | ❌ pending
nuScenes | VoxelNeXt     | Pert    | ✅     | ✅    | ❌ pending
nuScenes | TransFusion-L | Inj     | ✅     | ✅    | ❌ pending
nuScenes | TransFusion-L | Pert    | ✅     | ✅    | ❌ pending
```

**Dropping 行的处理**: 不放入主表。在正文中用 1-2 句话说明 "对于 dropping 攻击，M2 检测到密度下降模式后选择不干预，避免二次伤害"。这比放一行 "ASAP = No-defense" 更好。

### 2.3 与 LiDAR-SPD 的直接对比

LiDAR-SPD 是 ICASSP 2025 论文，同一会议的直接前作。**必须**在主表中加入 LiDAR-SPD 作为 baseline。

问题: 我们没有 LiDAR-SPD 的代码/checkpoint。

解决方案 (按优先级):
1. **直接引用其论文数字** — LiDAR-SPD Table I/II/III 报告了 KITTI PointPillars/PV-RCNN/CenterPoint 的 ASR。我们可以直接引用 (标注 "reported in [X]")。
2. **复现其核心机制** — 固定半径 SPU + 球形投影 + 全 SPU 扩散。这实际上就是我们的 "Uniform SPU diffusion (E3.3)" baseline 的一个变体。

**建议**: 引用 LiDAR-SPD 论文数字 + 跑 E3.3 Uniform SPU 作为公平消融对照。

---

## 3. 实验难度: 风险评估与降级策略

### 3.1 nuScenes ASAP 的技术风险

这是当前最大的 blocker。风险点：

| 风险 | 严重度 | 缓解策略 |
|------|--------|---------|
| Score-net 在 nuScenes 上不泛化 | 高 | 方案 A: 直接用 KITTI checkpoint (跨数据集泛化本身是 contribution); 方案 B: 在 nuScenes clean patches 上快速 finetune |
| M2 阈值 tau 需要重新校准 | 中 | 用 nuScenes 的 20 帧 attacked/clean 对做 ROC，复用 KITTI 的校准流程 |
| nuScenes 32-beam 360° 密度分布不同 | 中 | M1 自适应半径本身就是为了处理密度差异，理论上应该自动适配 |
| 计算时间过长 (6019 val frames) | 中 | 只跑 accepted attacks (Injection + Perturbation ε=0.5)，双卡并行 |

### 3.2 降级策略 (如果 nuScenes ASAP 效果不好)

如果 KITTI-trained score-net 在 nuScenes 上效果差于 ROR：

**Plan B**: 论文改为 "KITTI-focused + nuScenes transferability study"
- 主表只放 KITTI 4 行 (2 检测器 × 2 攻击)
- nuScenes 作为 "cross-dataset transferability" 讨论，即使效果不如 ROR 也可以诚实报告
- 这仍然比 LiDAR-SPD (只有 KITTI) 更广

**Plan C**: 如果连 Plan B 都不够
- 在 nuScenes 上用 local-mean fallback 而非 score-net
- 论文中说明 "score-net 在 KITTI 上训练，nuScenes 使用 local-mean 作为 M3 的轻量替代"
- 这仍然保留了 M1 + M2 的 contribution

### 3.3 消融实验的难度排序

| 消融 | 难度 | GPU 时间 | 信息量 |
|------|------|---------|--------|
| A3: selective vs uniform | 低 | ~2h (只需跑 uniform 一次) | **最高** — 直接证明 M2 gating 的价值 |
| A1: adaptive vs fixed radius | 低 | ~4h (4 个固定半径) | 高 — 证明 M1 的价值 |
| A4: tau 敏感性 | 低 | ~3h (5 个 tau 值) | 中 — 证明 tau 不需要精调 |
| A2: 单特征消融 | 中 | ~6h (4 个单特征) | 中 — 证明 4 特征组合优于单特征 |

---

## 4. 实验深度: 每个实验的具体执行方案

### 4.1 P0: nuScenes ASAP (E4.3 + E4.4)

**目标**: 填充主表 nuScenes 列

**执行步骤**:

```
Step 1: nuScenes M2 校准 (1-2 小时)
  - 从 nuScenes val 中选 20 帧 clean + 对应 attacked
  - 跑 M1 + M2 特征提取
  - 拟合 logistic + Youden-J 得到 tau_nuscenes
  - 验证: tau_nuscenes 与 tau_kitti 的差异

Step 2: nuScenes ASAP purification (预计 8-12 小时, 双卡)
  - 策略决策: 先用 KITTI score-net checkpoint 直接跑
  - 如果效果差, 再考虑 finetune 或 local-mean fallback
  - 只跑 accepted attacks: E2.1 Injection + E2.2 Perturbation(ε=0.5)
  - 命令模板:
    GPUS=0,1 WORKERS_PER_GPU=2 \
    bash scripts/run_asap_vpsde_score_net_dual_gpu.sh \
      checkpoints/kitti/asap_score_net_trackA_v1.pth \
      E2.1_injection_nuscenes

Step 3: nuScenes eval (预计 4-6 小时)
  - VoxelNeXt on purified E2.1 + E2.2
  - TransFusion-Lidar on purified E2.1 + E2.2
  - 解析 JSON summary

Step 4: 判断结果
  - 如果 ASAP > ROR: 直接写入主表
  - 如果 ASAP ≈ ROR: 仍然写入, 强调 cross-dataset zero-shot
  - 如果 ASAP < ROR: 执行 Plan B/C
```

**预计总时间**: 2-3 天 (含等待 GPU)

### 4.2 P0: 消融 A3 — Selective vs Uniform (E3.3 + E5.3)

**目标**: 证明 M2 gating 是关键

**执行步骤**:

```
Step 1: 实现 Uniform SPU purification
  - 修改 pipeline 参数: --tau 0.0 (所有 SPU 都扩散)
  - 或者: --force_purify_all true
  - 只需跑 KITTI E2.2 Perturbation (最能体现差异的攻击)

Step 2: 评估
  - PointPillars on uniform-purified E2.2
  - 对比: No-def (56.54) vs Uniform vs ASAP (61.04)

Step 3: 预期结果
  - Uniform 应该 < ASAP (因为过度净化损害 benign geometry)
  - Uniform 可能 > No-def (因为扩散本身有去噪效果)
  - 如果 Uniform > ASAP: 说明 M2 gating 有问题, 需要诊断
```

**预计时间**: 3-4 小时

### 4.3 P1: 推理延迟 (E6)

**目标**: 填充 Table 3

**执行步骤**:

```
Step 1: 测量 ASAP pipeline 各阶段延迟
  - 从现有 meta.jsonl 中提取 per-frame timing
  - 分解为: M1 (KNN + radius) + M2 (scoring) + M3 (diffusion)
  - 报告: mean ± std over 500 frames

Step 2: 测量 detector 延迟
  - PointPillars forward pass (已知约 20-30ms on T4)
  - 从 OpenPCDet eval log 中提取

Step 3: 测量 baseline 延迟
  - SOR/ROR: 从 defense pipeline log 提取
  - 如果没有 timing, 跑 100 帧计时

Step 4: 组装 Table 3
  - 关键数字: ASAP selective ratio |C*|/|C|
  - 从 meta.jsonl 的 flagged_spu_rate 字段提取
```

**预计时间**: 2-3 小时 (主要是解析已有 log)

### 4.4 P1: 消融 A1 — Adaptive vs Fixed Radius

**执行步骤**:

```
Step 1: 跑 4 个固定半径
  - r = 0.10, 0.20, 0.30, 0.40 m
  - 只跑 KITTI E2.2 Perturbation + PointPillars
  - 命令: python -m asap.pipeline --r_min X --r_max X (强制固定)

Step 2: 对比
  - 预期: 没有单一固定半径能同时在近处和远处都好
  - adaptive (当前默认) 应该是最优或接近最优
```

**预计时间**: 4-5 小时

---

## 5. 实验执行时间线

### 最优路径 (假设 nuScenes ASAP 成功)

```
Day 1 (今天):
  ├── [并行] nuScenes M2 校准 (2h)
  └── [并行] E3.3 Uniform SPU baseline on KITTI (3h)

Day 2:
  ├── nuScenes ASAP purification 启动 (overnight, 双卡)
  └── E6 延迟测量 (从已有 log 解析, 2h)

Day 3:
  ├── nuScenes ASAP purification 完成
  ├── nuScenes eval (VoxelNeXt + TransFusion, 4-6h)
  └── [并行] A1 fixed radius 消融 (4h)

Day 4:
  ├── 所有数字就位
  ├── 填充论文 Table 1 + Table 2 + Table 3
  └── 更新 abstract/conclusion 中的数字

Day 5-7:
  ├── 论文修改 (method/experiments/related work)
  ├── /ars-reviewer 内部审稿
  └── LaTeX 排版
```

### 降级路径 (如果 nuScenes ASAP 效果差)

```
Day 1-2: 同上
Day 3: 发现 nuScenes 效果差
  ├── 尝试 local-mean fallback (2h)
  ├── 如果 fallback 也差: 执行 Plan B
  └── 调整论文 claim 为 "KITTI-focused + transferability study"
Day 4-5: 重写实验部分
Day 6-7: 排版
```

---

## 6. 论文中需要诚实讨论的点

### 6.1 必须讨论 (reviewer 一定会问)

1. **Dropping 为什么 skip**: "M2 检测到 dropping 模式 (密度下降但几何结构保持) 后，选择不干预。对 dropping 攻击，任何基于点重建的防御都可能引入更多误差。这是一个保守但安全的设计选择。"

2. **PV-RCNN Injection: ROR 略优**: "在 PV-RCNN Injection 上，ROR (72.77) 与 ASAP (72.59) 接近。ASAP 的优势在于跨攻击一致性: ROR 在 Perturbation 上完全失效 (1.94)，而 ASAP 仍有改善 (10.96)。"

3. **PV-RCNN Perturbation 恢复有限**: "当前 geometric perturbation 对 PV-RCNN 的破坏极其严重 (3.50 AP)，所有防御方法恢复都有限。这反映了 voxel-point 两阶段架构对坐标扰动的高敏感性。"

4. **Score-net 训练规模**: "当前 score-net 在 16384 KITTI patches 上训练 20 epochs。更大规模训练可能进一步提升效果，但当前结果已经证明了选择性扩散的有效性。"

### 6.2 可选讨论 (如果空间允许)

- 攻击强度: 当前使用 geometric/random 攻击而非 gradient-based。这是 limitation，但也意味着 ASAP 不需要假设攻击者的知识。
- nuScenes Dropping 失败: keyframe-only 攻击在 10-sweep 评估下太弱。

---

## 7. 与 LiDAR-SPD 的对比策略

LiDAR-SPD 是 ICASSP 2025 论文 (同一会议前一年)，必须正面对比。

### 7.1 可以直接引用的 LiDAR-SPD 数字

从 LiDAR-SPD Table I (PointPillars, ASR%):

| Attack | 3D-VField | Hahner | PointDP | **LiDAR-SPD** |
|--------|-----------|--------|---------|---------------|
| Wang [15] | 77.6 | 71.2 | 54.9 | **25.1** |
| Sun [13] | 75.8 | 73.3 | 70.8 | **20.5** |
| Tu [14] | 70.3 | 71.4 | 60.2 | **40.7** |
| Wang [16] | 95.4 | 94.1 | 93.0 | **46.6** |

从 LiDAR-SPD Table II (PV-RCNN, ASR%):

| Attack | 3D-VField | Hahner | PointDP | **LiDAR-SPD** |
|--------|-----------|--------|---------|---------------|
| Wang [15] | 72.5 | 66.3 | 47.6 | **19.4** |
| Sun [13] | 74.9 | 72.4 | 68.6 | **14.5** |
| Tu [14] | 64.0 | 66.8 | 54.1 | **35.2** |
| Wang [16] | 95.1 | 94.6 | 93.8 | **47.2** |

### 7.2 对比注意事项

- LiDAR-SPD 使用的攻击 (Wang/Sun/Tu) 与我们的 geometric 攻击**不完全相同**
- 不能直接把 LiDAR-SPD 的 ASR 数字放入我们的表格 (攻击设置不同)
- **正确做法**: 在 related work 中引用其数字，在 experiments 中说明 "我们的 Uniform SPU baseline 近似 LiDAR-SPD 的全 SPU 扩散策略"

### 7.3 E3.3 作为 LiDAR-SPD 的公平近似

跑 E3.3 (Uniform SPU diffusion) 时，使用:
- 固定半径 r₁ = 0.15m (LiDAR-SPD 的设置)
- 所有 SPU 都扩散 (无 M2 gating)
- 同样的 VP-SDE score-net

这给出一个公平的 "LiDAR-SPD-style" baseline，在我们的攻击设置下评估。

---

## 8. 总结: 确保投稿顺利的关键检查清单

### 必须完成 (投稿 hard gate)

- [ ] nuScenes ASAP 至少 2 行有数字 (VoxelNeXt Injection + Perturbation)
- [ ] 消融 A3 (selective vs uniform) 有结果
- [ ] Table 3 延迟数字填入
- [ ] references.bib 所有 UNVERIFIED 标记清除
- [ ] 论文中无 [XX.X] 占位符
- [ ] 所有 claim 不超过实验支撑

### 强烈建议 (提升接收概率)

- [ ] nuScenes TransFusion-Lidar ASAP 行也有数字
- [ ] 消融 A1 (adaptive vs fixed) 有结果
- [ ] E3.3 Uniform SPU 作为 LiDAR-SPD 近似 baseline
- [ ] 1 张可视化图 (purification before/after)

### 可选 (如果时间充裕)

- [ ] 消融 A2 (单特征)
- [ ] 消融 A4 (tau 敏感性)
- [ ] Score-net v2 (更大训练集)
