# E3 — Baseline defenses

Non-ASAP defenses that we compare against. Each file documents the defense pipeline, hyperparameters, and the per-detector / per-attack evaluation procedure.

| ID | Defense | Role | File | Status |
|----|---------|------|------|--------|
| E3.1 | SOR (Statistical Outlier Removal)        | Classical denoising baseline                | [`E3.1_sor.md`](E3.1_sor.md)                                       | **done for KITTI/PointPillars E2.1-E2.3** |
| E3.2 | ROR (Radius Outlier Removal)             | Classical denoising baseline                | [`E3.2_ror.md`](E3.2_ror.md)                                       | **done for KITTI/PointPillars E2.1-E2.3** |
| E3.3 | Uniform SPU diffusion (ASAP minus M2)    | Internal ablation and LiDAR-SPD-style fixed-radius proxy | [`E3.3_uniform_spu_diffusion.md`](E3.3_uniform_spu_diffusion.md)   | **done: r=0.40 and LiDAR-SPD-style r=0.15 proxies** |
| E3.4 | External published purifier              | External reference (e.g. Ada3Diff)           | [`E3.4_external_published.md`](E3.4_external_published.md)         | optional |

Why each is here:
- **E3.1 / E3.2** isolate the value of the diffusion step in ASAP (M3).
- **E3.3** isolates the value of the anomaly-guided selection (M2 + threshold $\tau$).
- **E3.4** provides an external comparison point when checkpoints are available, but is optional given ICASSP's 4-page budget.

Current output layout:

```text
outputs/defenses/kitti/E3.1_sor/E2.{1,2,3}_*/velodyne_defended/
outputs/defenses/kitti/E3.2_ror/E2.{1,2,3}_*/velodyne_defended/
```

The evaluated SOR/ROR metrics are currently consolidated in `docs/experiments/E4_main_results/E4.1_kitti_pointpillars.md` and `outputs/asap_vpsde/kitti_pointpillars_eval_summary.json`.
