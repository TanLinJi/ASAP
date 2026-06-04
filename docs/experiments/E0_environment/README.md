# E0 — Environment

Setup, data, and pretrained-weight preparation. Nothing in this family produces a paper number on its own; everything here is a prerequisite for `E1..E6`.

| ID | Name | File | Status |
|----|------|------|--------|
| E0.1 | KITTI 3D object detection data preparation | [`E0.1_kitti_data_preparation.md`](E0.1_kitti_data_preparation.md) | **done** |
| E0.2 | `asap` conda environment                  | [`E0.2_asap_conda_env.md`](E0.2_asap_conda_env.md)                 | **done** |
| E0.3 | OpenPCDet backend + KITTI infos           | [`E0.3_openpcdet_kitti_infos.md`](E0.3_openpcdet_kitti_infos.md)   | **done** |
| E0.4 | Detector pretrained checkpoints           | [`E0.4_detector_checkpoints.md`](E0.4_detector_checkpoints.md)     | **done** — 4/4 checkpoints end-to-end verified bit-exact: PointPillars 77.28 (E1.1), PV-RCNN 83.69 (E1.2), VoxelNeXt mAP 60.52 / NDS 66.64 (E1.3), TransFusion-L mAP 64.57 / NDS 69.43 (E1.4). All four cells of paper Table 1 / *Clean (ref)* are locked in. |
| E0.5 | nuScenes data preparation                 | [`E0.5_nuscenes_data_preparation.md`](E0.5_nuscenes_data_preparation.md) | **done** — `v1.0-mini` smoke 2026-05-21; `v1.0-trainval` downloaded 14:53→17:49 (293 GiB), extracted 19:10→20:42 (92 min), `create_nuscenes_infos --version v1.0-trainval` finished 2026-05-22 00:16 (`_infos_train.pkl` 423 MB / `_infos_val.pkl` 83 MB / `_dbinfos_*.pkl` 190 MB / gt_database 823k bins). Canonical 850-scene/34149-sample/1.17M-annotation split preserved. |

Downstream consumers:
- `E1.*` need `E0.1`, `E0.2`, `E0.3`, `E0.4`.
- `E1.3`, `E1.4` additionally need `E0.5`.
- All `E4.*`, `E5.*`, `E6.*` inherit the same chain transitively.
