# E4 — Main results

Each E4 experiment fills one row block of the main detector-agnostic table in `paper/04_experiments.md` (Table 1). The cell value is `(detection metric, ASR)` per `(attack, defense)` cross-product.

| ID | Dataset      | Detector             | Detector family                                | File | Status |
|----|--------------|----------------------|------------------------------------------------|------|--------|
| E4.1 | KITTI val    | PointPillars         | Pillar-based one-stage                         | [`E4.1_kitti_pointpillars.md`](E4.1_kitti_pointpillars.md)                       | **Track A policy candidate validated** |
| E4.2 | KITTI val    | PV-RCNN              | Voxel-point two-stage                          | [`E4.2_kitti_pvrcnn.md`](E4.2_kitti_pvrcnn.md)                                   | **done** — revised policy + SOR/ROR baselines |
| E4.3 | nuScenes val | VoxelNeXt (0.075)    | Fully sparse voxel CNN, anchor-free            | [`E4.3_nuscenes_voxelnext.md`](E4.3_nuscenes_voxelnext.md)                       | **accepted rows complete; ASAP-pfilter validated** |
| E4.4 | nuScenes val | TransFusion-Lidar    | Sparse voxel backbone + transformer query head | [`E4.4_nuscenes_transfusion_lidar.md`](E4.4_nuscenes_transfusion_lidar.md)       | **accepted rows complete; ASAP-pfilter transfer validated** |
| E4.5 | Waymo val    | MPPNet (temporal)    | Multi-frame transformer                        | [`E4.5_deferred_waymo_mppnet.md`](E4.5_deferred_waymo_mppnet.md)                 | **deferred** — out of ICASSP scope |

> **Scope change (2026-05-21).** `E4.3` / `E4.4` migrated from Waymo (DSVT, VoxelNeXt-Waymo) to nuScenes (VoxelNeXt-NuScenes, TransFusion-Lidar). `E4.5` (Waymo / MPPNet) is *deferred* and kept on file as a documented future-work hook with explicit re-activation criteria. See `E0.5` §3.0 for the decision record.

Main accepted row structure:

| Attack | Main handling |
|--------|---------------|
| Injection (E2.1) | Report no-defense, SOR/ROR, and ASAP-v1. ROR remains marginally stronger on some detector rows, so phrase ASAP-v1 as near-ROR recovery rather than dominance. |
| Perturbation (E2.2) | Report accepted nuScenes `epsilon=0.5` rows and explicitly tag the revised cascade as `ASAP-pfilter`. |
| Dropping (E2.3) | Report KITTI conservative skip-M3 behavior; keep nuScenes Dropping excluded because keyframe-only dropping fails the attack-strength gate. |

E4.1 already has a richer diagnostic matrix in `E4.1_kitti_pointpillars.md`: no-defense, SOR, ROR, ASAP-drop, ASAP-vpsde local-mean fallback, hybrid ablations, pure score-net, score-net+anchor b075, and the Track A policy. E4.2 confirms that the Injection branch transfers to PV-RCNN and motivates the revised policy: keep learned M3 for Injection/Perturbation, but skip M3 for Dropping because the current dropping attack removes evidence and learned editing compounds the damage. E4.3/E4.4 use a low-cost nuScenes keyframe-only attack protocol: current keyframes are edited, historical sweeps remain clean. Accepted nuScenes attacks are Injection with the original budget and Perturbation with `epsilon=0.5`. Dropping is paused after failing the keyframe-only attack-strength gate even at `drop_ratio=1.0`. The final paper table should not claim per-row dominance: ROR remains marginally stronger on Injection, while ASAP-pfilter is stronger on Perturbation across both nuScenes detectors.

The current E4 values have been aggregated into:
- `paper/04_experiments.md` Table 1 (main),
- `paper/00_abstract.md`,
- `paper/05_conclusion.md`.
