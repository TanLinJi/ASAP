# Detector selection for ASAP experiments

This note records the detector choices used to support ASAP's **detector-agnostic** claim.

## Key conclusion

ASAP should not claim detector-agnostic robustness based only on KITTI + PointPillars. PointPillars is a valid LiDAR 3D detector, but it is an older pillar-based one-stage baseline. A detector-agnostic defense claim needs evidence across multiple detector families and, ideally, recent strong detectors.

## Evidence from the DSVT paper

The DSVT paper, **"DSVT: Dynamic Sparse Voxel Transformer With Rotated Sets"** (CVPR 2023), compares against strong detectors mainly on **Waymo Open Dataset** and **nuScenes**, rather than using KITTI as the main SOTA benchmark. The paper highlights the field's shift from KITTI to larger-scale benchmarks for modern detector comparison.

Important detectors mentioned in the DSVT comparison include:

| Detector | Year / venue in DSVT paper | Dataset focus | Notes for ASAP |
|----------|-----------------------------|---------------|----------------|
| CenterPoint-Voxel | CVPR 2021 | Waymo / nuScenes | Strong center-based voxel baseline, older but widely used. |
| SST | CVPR 2022 | Waymo | Sparse transformer detector baseline. |
| PillarNet | ECCV 2022 | Waymo | Strong pillar-style baseline. |
| CenterFormer | ECCV 2022 | Waymo | Multi-frame center-based transformer detector. |
| PV-RCNN++ | IJCV 2022 | Waymo / KITTI | Strong two-stage detector, but no longer very recent. |
| MPPNet | ECCV 2022 | Waymo | Strong multi-frame two-stage detector, ranks highly on Waymo. |
| DSVT | CVPR 2023 | Waymo / nuScenes | Dynamic Sparse Voxel Transformer; recent and strong. |
| VoxelNeXt | CVPR 2023 | nuScenes / Waymo / Argoverse2 | Fully sparse voxel detector; recent and strong. |
| TransFusion-Lidar | CVPR 2022 | nuScenes | Strong LiDAR-only transformer-style head. |
| BEVFusion | NeurIPS 2022 | nuScenes | Multi-modal BEV detector; useful only if ASAP is evaluated with image-LiDAR inputs. |

## OpenPCDet support in the local backend

The local OpenPCDet backend contains configs for several recent or strong detectors:

| Detector | Local config | Recommended ASAP role |
|----------|--------------|-----------------------|
| DSVT-Pillar | `tools/cfgs/waymo_models/dsvt_pillar.yaml` | Main modern detector for Waymo. |
| DSVT-Voxel | `tools/cfgs/waymo_models/dsvt_voxel.yaml` | Main modern detector for Waymo. |
| VoxelNeXt | `tools/cfgs/waymo_models/voxelnext_ioubranch_large.yaml` | Modern sparse voxel detector on Waymo. |
| VoxelNeXt-2D | `tools/cfgs/waymo_models/voxelnext2d_ioubranch.yaml` | Optional Waymo comparison. |
| MPPNet | `docs/guidelines_of_approaches/mppnet.md` | Strong multi-frame Waymo detector, but heavier to reproduce. |
| TransFusion-Lidar | `tools/cfgs/nuscenes_models/transfusion_lidar.yaml` | Main LiDAR-only nuScenes detector. |
| BEVFusion | `tools/cfgs/nuscenes_models/bevfusion.yaml` | Optional multi-modal detector; only use if ASAP is extended to preserve image-LiDAR pipeline assumptions. |
| VoxelNeXt on nuScenes | `tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml` | Modern nuScenes LiDAR-only detector. |
| PV-RCNN++ | `tools/cfgs/kitti_models/pv_rcnn_plusplus_reproduced_by_community.yaml` and Waymo configs | Useful bridge baseline, but not the main SOTA evidence. |

## Recommended experimental positioning

### Main claim experiments

Use **Waymo** and/or **nuScenes** with modern detectors:

- **Waymo**: DSVT-Voxel, VoxelNeXt, MPPNet.
- **nuScenes**: VoxelNeXt and TransFusion-Lidar.

These experiments best support the detector-agnostic claim because they cover different detector families:

- transformer-style sparse voxel backbone: DSVT;
- fully sparse voxel detector: VoxelNeXt;
- temporal multi-frame detector: MPPNet;
- transformer/query-based detection head: TransFusion-Lidar.

### Fast reproducibility experiments

Use **KITTI** for early development and sanity checks:

- PointPillars: fast classical baseline;
- PV-RCNN++: stronger KITTI baseline available in OpenPCDet;
- Voxel R-CNN or SECOND: optional architecture diversity.

KITTI results should be framed as a reproducibility and ablation setting, not the main evidence for recent SOTA detector-agnostic robustness.

## Abstract wording guideline

Avoid listing only old KITTI detectors in the Abstract. Prefer this wording style:

> We evaluate ASAP across modern LiDAR detectors on KITTI, Waymo, and nuScenes, including transformer-style sparse voxel detectors and fully sparse voxel detectors such as DSVT and VoxelNeXt.

Until all experiments are available, keep numerical claims as placeholders and avoid overclaiming.
