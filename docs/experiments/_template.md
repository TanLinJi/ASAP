# E<id> — <short experiment name>

<!--
模板用法：
- 复制此文件到 docs/experiments/<family>/E<id>_<short_name>.md。
- 把 <id> 换成实验编号（如 E0.1、E4.3）；<short experiment name> 换成实验短名。
- 凡是 [...] 占位都需要在执行前/执行中填写。
- 字段尽量保留，不要删；可以填 "n/a"。
-->

| Field | Value |
|-------|-------|
| **Experiment ID** | `E<id>` |
| **Family**        | `E<family>` (e.g. `E0_environment`, `E4_main_results`) |
| **Short name**    | `<short_name>` |
| **Owner**         | <responsible person> |
| **Status**        | `pending` / `running` / `done` / `blocked` / `deprecated` |
| **Last update**   | `YYYY-MM-DD` |
| **Linked paper section** | e.g. `paper/04_experiments.md` Table 1, row 3 |
| **Linked code / configs** | absolute paths to scripts, configs, logs |

---

## 1. Purpose / hypothesis

What is this experiment supposed to demonstrate or measure? State the hypothesis in one sentence, and the *exact* question the result should answer. For ablations, also state the *null* outcome (what would make us reject the hypothesis).

## 2. Dependencies

List every prerequisite experiment / artifact, by `E<id>` and absolute path. Examples:

- Upstream experiments: `E0.3` (KITTI infos), `E0.4` (PointPillars checkpoint).
- Upstream artifacts: `/root/autodl-tmp/ASAP/data/kitti/kitti_infos_val.pkl`, `/root/autodl-tmp/ASAP/checkpoints/<dataset>/<name>.pth`.
- Upstream code/config: `third_party/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml`.

If anything is missing, mark this experiment as `blocked` and link the blocker.

## 3. Setup

### 3.1 Hardware & environment

- Machine: <node name>, GPUs: <model x count>, driver / CUDA version.
- Conda env: `asap` (Python 3.9, PyTorch 2.5.1+cu124, spconv 2.3.6).
- Relevant environment variables (`LD_LIBRARY_PATH`, `PYTHONPATH`, etc.).

### 3.2 Data

- Dataset(s): KITTI / Waymo / nuScenes / synthetic.
- Split: `train` / `val` / `test`, frame count, sample indices if subset.
- Pre-processing / attack pipeline applied (link to the attack experiment).

### 3.3 Detector(s) / defense(s)

- Detector(s): name, config path, checkpoint path, version.
- Defense(s): ASAP variant / SOR / ROR / uniform SPU diffusion / no-defense.
- Method hyperparameters (M1 `alpha`, `k`, `r_min`, `r_max`, `beta`; M2 logistic weights, `tau`; M3 schedule, `t_star`, batch size).

### 3.4 Reproducible command

Always paste the **exact** command(s) that produced the recorded results:

```bash
# example
cd /root/autodl-tmp/ASAP/third_party/OpenPCDet
LD_LIBRARY_PATH=/root/miniconda3/envs/asap/lib/python3.9/site-packages/torch/lib:/usr/local/cuda-12.4/lib64 \
  conda run -n asap python tools/test.py \
    --cfg_file tools/cfgs/kitti_models/pointpillar.yaml \
    --ckpt /root/autodl-tmp/ASAP/checkpoints/<dataset>/<ckpt>.pth \
    --batch_size 4
```

## 4. Procedure

Numbered step-by-step description of what was done. One step per Markdown list item. Include both what you ran and what you observed.

1. ...
2. ...
3. ...

## 5. Expected output

- Files: log paths, result `.pkl`, generated `.bin`, figure paths.
- Metrics: which numbers should appear, in what unit, with what precision.

## 6. Actual results

Free-form section with:

- Headline numbers in a small Markdown table.
- Raw log excerpts inside fenced code blocks.
- Links (absolute paths) to full log files and figures.

| Metric | Value | Unit | Notes |
|--------|-------|------|-------|
|        |       |      |       |

## 7. Analysis

Short prose interpretation:

- Does the result confirm or reject the hypothesis from Section 1?
- How does it compare to the expected range / prior work?
- Any surprising patterns, edge cases, or failure modes?
- Implications for downstream experiments and for the paper claims.

## 8. Issues encountered & resolutions

- Issue 1: description; root cause; fix.
- Issue 2: ...

## 9. Next actions

- Things to follow up on (per-experiment).
- Cross-links to dependent experiments.

## 10. Change log

| Date | Author | Change |
|------|--------|--------|
| YYYY-MM-DD | <name> | Created skeleton |
