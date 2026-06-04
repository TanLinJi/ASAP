# E5 — Ablations

Module-level ablations of ASAP. To keep the design space tractable, **all E5 experiments use KITTI val + PointPillars** unless explicitly noted in the file's *Setup* section. They populate Table 2 in `paper/04_experiments.md`.

| ID | Target module | Description | File | Status |
|----|---------------|-------------|------|--------|
| E5.1 | M1 Adaptive Radius   | Fixed $r_1 \in \{0.10, 0.20, 0.30, 0.40\}$ m vs density-adaptive rule | [`E5.1_adaptive_radius.md`](E5.1_adaptive_radius.md) | **done** |
| E5.2 | M2 Anomaly Scorer    | Lightweight feature diagnostic plus detector validation for `only_f_d` / `drop_f_d` | [`E5.2_anomaly_features.md`](E5.2_anomaly_features.md) | **done** |
| E5.3 | M3 Selective Diffusion / purifier variants | Uniform SPU diffusion (no $\tau$ gating) vs full ASAP; current evidence also includes drop/local-mean/hybrid variants from E4.1 | [`E5.3_selective_diffusion.md`](E5.3_selective_diffusion.md) | **done for paper Table 2** |
| E5.4 | Threshold $\tau$     | Focused sweep around the ROC-calibrated operating point       | [`E5.4_threshold_tau.md`](E5.4_threshold_tau.md) | **done** |
| E5.5 | Qualitative selectivity | BEV point-cloud comparisons and per-frame edit/drop statistics for representative KITTI and nuScenes rows | [`E5.5_qualitative_selectivity.md`](E5.5_qualitative_selectivity.md) | **done** |
| E5.6 | Track B PV-RCNN diagnostic | KITTI/PV-RCNN Perturbation pfilter, blend, and `t_star` tests | [`E5.6_kitti_pvrcnn_trackB_pfilter.md`](E5.6_kitti_pvrcnn_trackB_pfilter.md) | **done — `t_star=0.08` positive; `0.10` not promoted** |

Reading guideline:
- Each ablation isolates one module while keeping the other two at the defaults defined in `paper/03_method.md` §3.3.4, §3.4.5, §3.5.
- Report both detection metric (3D AP on KITTI Car) and ASR; the goal is to show that **removing any module hurts at least one of the two**.
- The paper-facing ablation set is complete for the current claim boundary. Any new E5 run should target a specific reviewer risk rather than broaden the sweep.
