# E1 — Clean baselines

Clean-scene detection numbers, one experiment per `(dataset, detector)` pair. These are the *upper-bound reference rows* used by every E4 main table and every E5 ablation.

| ID | Dataset      | Detector             | Metric                | File | Status |
|----|--------------|----------------------|-----------------------|------|--------|
| E1.1 | KITTI val    | PointPillars         | 3D AP @ Car/Ped/Cyc   | [`E1.1_kitti_pointpillars.md`](E1.1_kitti_pointpillars.md)               | **done** — matches OpenPCDet to ±0.07 |
| E1.2 | KITTI val    | PV-RCNN              | 3D AP @ Car/Ped/Cyc   | [`E1.2_kitti_pvrcnn.md`](E1.2_kitti_pvrcnn.md)                           | **done** — Car/Ped/Cyc 3D AP@R11 Mod = 83.69 / 54.84 / 69.48; *bit-exact* match to ckpt filename `pv_rcnn_8369.pth` |
| E1.3 | nuScenes val | VoxelNeXt (0.075)    | mAP / NDS             | [`E1.3_nuscenes_voxelnext.md`](E1.3_nuscenes_voxelnext.md)               | **done** — trainval val (6019 samples) mAP 0.6052 / NDS 0.6664; Δ vs OpenPCDet README -0.0001 / -0.0001 (bit-exact) |
| E1.4 | nuScenes val | TransFusion-Lidar    | mAP / NDS             | [`E1.4_nuscenes_transfusion_lidar.md`](E1.4_nuscenes_transfusion_lidar.md) | **done** — trainval val (6019 samples) mAP 0.6457 / NDS 0.6943; Δ vs OpenPCDet README -0.0001 / **0.0000** (NDS bit-exact) |

> **Scope change (2026-05-21).** `E1.3` / `E1.4` migrated from Waymo (DSVT, VoxelNeXt-Waymo) to nuScenes (VoxelNeXt-NuScenes, TransFusion-Lidar). See `E0.5` §3.0 for the decision record.

Rules:
- All clean baselines run with **official pretrained checkpoints from OpenPCDet**; no detector is retrained for ASAP.
- The number reported by E1.x is the "Clean (ref)" column in `paper/04_experiments.md` Table 1.
- Re-run E1.x if any dataset or detector config is updated, and bump the change log inside the file.
