# ASAP implementation plan

This document records the concrete implementation plan for ASAP: Adaptive Spherical Anomaly-Guided Purification Against LiDAR Point Cloud Attacks.

## 1. Current local status

### Project boundary

- ASAP project root: `/root/autodl-tmp/ASAP`
- ASAP is treated as a new and independent project.
- Legacy local directories, especially `/root/autodl-tmp/LiDAR_SPD/`, are not part of ASAP.
- If an old dependency is still useful, re-clone it from its upstream Git repository under `/root/autodl-tmp/ASAP/third_party/`, remove the cloned project's inner `.git` directory, preserve its license file, and document its copyright and provenance in `third_party/README.md`.
- The preferred detector backend location is therefore `/root/autodl-tmp/ASAP/third_party/OpenPCDet/`, created from a fresh upstream clone when needed.

### Existing data and weights

- No real KITTI point cloud data has been placed under `/root/autodl-tmp/ASAP/data/kitti/` yet.
- KITTI point clouds, labels, calibration files, and images should be placed under the ASAP data directory and must not be committed to Git.
- Detector checkpoints should be placed under `/root/autodl-tmp/ASAP/checkpoints/` or `/root/autodl-tmp/ASAP/weights/`, and must not be committed to Git.
- MMDetection3D checkpoints should not be mixed directly with OpenPCDet unless we intentionally switch the detector backend.

## 2. Which KITTI data is needed

We need the official KITTI 3D object detection dataset, not KITTI raw, not KITTI odometry, and not SemanticKITTI.

The correct benchmark is usually listed as:

- KITTI Vision Benchmark Suite
- Object Detection Evaluation
- 3D Object Detection benchmark

### Minimum data for our first stage

For clean validation mAP and attack or defense experiments on the standard KITTI train/val split, we only need the official training part with labels:

```text
kitti/
├── ImageSets/
│   ├── train.txt
│   ├── val.txt
│   ├── trainval.txt
│   └── test.txt
└── training/
    ├── calib/      # camera-LiDAR calibration, required
    ├── image_2/    # left camera images, strongly recommended because OpenPCDet expects them
    ├── label_2/    # 3D detection labels, required for validation and training infos
    └── velodyne/   # LiDAR point clouds, required
```

Required official download items:

- Velodyne point clouds
- Camera calibration matrices
- Training labels
- Left color images

### Optional data

The official testing split is not needed for the first research loop because it has no public labels. It is only needed later if we submit results to the KITTI test server.

```text
kitti/
└── testing/
    ├── calib/
    ├── image_2/
    └── velodyne/
```

Optional OpenPCDet extras:

- `planes/`: road plane files, optional for some training augmentations.
- `depth_2/`: only needed for image-depth based methods such as CaDDN. We do not need it for PointPillars, PV-RCNN, CenterPoint, or ASAP's first implementation.

### Recommended local placement

Use the ASAP project data layout:

```text
/root/autodl-tmp/ASAP/data/kitti/
├── ImageSets/
├── training/
│   ├── calib/
│   ├── image_2/
│   ├── label_2/
│   └── velodyne/
└── testing/                 # optional for now
    ├── calib/
    ├── image_2/
    └── velodyne/
```

The KITTI files should live under the ASAP project directory, but they must not be committed to Git. The repository `.gitignore` keeps `data/**` ignored while preserving `data/.gitkeep`.

OpenPCDet should be treated as an external detection backend. When OpenPCDet needs its conventional `data/kitti` path, expose the ASAP dataset to it by symlink or by an ASAP-specific dataset config, instead of storing the real data under any legacy project.

## 3. Stage A: prepare KITTI and OpenPCDet baseline

### A1. Put KITTI into OpenPCDet format

After downloading and extracting the official KITTI object detection files, verify these counts:

- `training/velodyne`: 7,481 `.bin` files
- `training/calib`: 7,481 `.txt` files
- `training/image_2`: 7,481 image files
- `training/label_2`: 7,481 `.txt` files

If the optional testing split is downloaded:

- `testing/velodyne`: 7,518 `.bin` files
- `testing/calib`: 7,518 `.txt` files
- `testing/image_2`: 7,518 image files

### A2. Generate OpenPCDet KITTI infos

Before generating infos, make the OpenPCDet backend see the ASAP dataset path. The preferred layout remains:

```text
/root/autodl-tmp/ASAP/data/kitti/
```

If using OpenPCDet unchanged, its `data/kitti` path should point to the ASAP dataset through a symlink:

```text
/root/autodl-tmp/ASAP/third_party/OpenPCDet/data/kitti
  -> /root/autodl-tmp/ASAP/data/kitti
```

Then run from `/root/autodl-tmp/ASAP/third_party/OpenPCDet`:

```bash
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

Expected generated files include:

- `/root/autodl-tmp/ASAP/data/kitti/kitti_infos_train.pkl`
- `/root/autodl-tmp/ASAP/data/kitti/kitti_infos_val.pkl`
- `/root/autodl-tmp/ASAP/data/kitti/kitti_infos_trainval.pkl`
- `/root/autodl-tmp/ASAP/data/kitti/kitti_dbinfos_train.pkl`
- `/root/autodl-tmp/ASAP/data/kitti/gt_database/`

If the testing split is present, `kitti_infos_test.pkl` can also be generated.

### A3. Download a KITTI PointPillars checkpoint

For the first baseline, use an official OpenPCDet KITTI PointPillars checkpoint matching:

```text
tools/cfgs/kitti_models/pointpillar.yaml
```

Save it under:

```text
/root/autodl-tmp/ASAP/checkpoints/openpcdet/
```

The checkpoint must match OpenPCDet's KITTI PointPillars config. NuScenes CenterPoint checkpoints or MMDetection3D-format checkpoints are not suitable for this first OpenPCDet KITTI baseline.

### A4. Run clean PointPillars validation

Run from `/root/autodl-tmp/ASAP/third_party/OpenPCDet`:

```bash
python tools/test.py \
  --cfg_file tools/cfgs/kitti_models/pointpillar.yaml \
  --ckpt /root/autodl-tmp/ASAP/checkpoints/openpcdet/<kitti_pointpillar_checkpoint>.pth \
  --batch_size 4
```

Goal:

- Obtain clean mAP on KITTI val split.
- mAP means mean Average Precision, i.e. average detection precision over the validation set.
- This is the clean reference for later attack and defense experiments.

## 4. Stage B: establish no-defense attack baselines

Before implementing ASAP, we need no-defense attack success numbers.

### B1. Decide attack protocol

For KITTI 3D detection, Attack Success Rate means the percentage of selected attack cases where the attack achieves its goal.

ASR = Attack Success Rate.

Recommended first definitions:

- Object hiding attack: success if the target car is no longer detected with sufficient 3D IoU and confidence.
- Object injection or spoofing attack: success if a false car-like detection appears in the target region.
- Point perturbation attack: success if small coordinate changes cause target detection failure or serious confidence drop.

For KITTI Car class, common 3D IoU threshold is 0.7.

### B2. Start with two attacks

First implement or adapt two representative attacks from the proposal:

1. Sun20-style point injection attack
2. Wang21-style point perturbation attack

The first implementation does not need to cover every detector. It should first work on PointPillars with a fixed validation subset.

### B3. Save attacked point clouds offline

For reproducibility, use an offline attacked dataset layout first:

```text
outputs/attacks/kitti_pointpillar/
├── sun20_injection/
│   ├── velodyne_attacked/
│   └── attack_meta.jsonl
└── wang21_perturbation/
    ├── velodyne_attacked/
    └── attack_meta.jsonl
```

This is slower than direct dataloader hooking but easier to debug.

## 5. Stage C: implement ASAP geometric frontend

ASAP should be implemented inside `/root/autodl-tmp/ASAP/src/asap/` and tested independently from OpenPCDet first.

### C1. Adaptive SPU builder

SPU means Spherical Purification Unit.

Input:

- point cloud array with shape `[N, 3]` or `[N, 4]`
- xyz coordinates and optional intensity

Core rule:

```text
r1 = alpha * d_k(p)
r1 is clipped into [r_min, r_max]
r2 = inner_ratio * r1
```

Default values from proposal:

- `alpha = 2.0`
- `k = 16`
- `r_min = 0.1 m`
- `r_max = 0.4 m`
- `inner_ratio = 0.67`

Deliverables:

- `src/asap/geometry/spu_builder.py`
- unit tests under `tests/`
- synthetic point cloud test cases for near, middle, and far range density

### C2. Spherical projection

Purpose:

- Remove suspicious injected points based on LiDAR occlusion geometry.
- In a small angular bin, nearer points should occlude farther inconsistent points.

Deliverables:

- `src/asap/geometry/spherical_projection.py`
- tests for injected outliers behind or in front of valid surfaces

### C3. Offline preprocessing interface

Before touching OpenPCDet internals, implement a simple file-level purifier:

```text
input KITTI velodyne .bin -> output purified .bin
```

Deliverables:

- `scripts/purify_kitti_bin.py`
- sample command documented in `docs/`

## 6. Stage D: implement anomaly scorer

The anomaly scorer should operate at SPU level.

Features:

- density: number of points per SPU volume or angular area
- compactness: how concentrated points are around the SPU center
- anisotropy: PCA eigenvalue ratio, where PCA means Principal Component Analysis
- vMF kappa: concentration of directions under von Mises-Fisher style directional statistics

Deliverables:

- `src/asap/scoring/features.py`
- `src/asap/scoring/anomaly_scorer.py`
- `src/asap/scoring/threshold.py`

Threshold calibration:

- ROC means Receiver Operating Characteristic.
- ROC helps choose threshold tau by comparing true suspicious SPUs and clean SPUs.
- The initial tau can be calibrated on synthetic attacks before full attack code is stable.

## 7. Stage E: selective diffusion interface

Do not start by training a diffusion model. First define the interface and use a safe identity purifier to make the pipeline testable.

### E1. Interface

Deliverables:

- `src/asap/diffusion/base.py`
- `src/asap/diffusion/identity.py`
- `src/asap/pipeline/purifier.py`

Behavior:

```text
if anomaly_score > tau:
    send SPU to diffusion purifier
else:
    bypass SPU
```

### E2. Real diffusion backend

After the geometric pipeline is stable, add PointDP-style or VP-SDE style diffusion purification.

VP-SDE means Variance Preserving Stochastic Differential Equation.

Deliverables:

- diffusion model adapter
- batch padding for suspicious SPUs
- runtime measurement

## 8. Stage F: connect ASAP with OpenPCDet evaluation

Use the simplest integration first:

1. Read original or attacked KITTI `.bin` files.
2. Purify and write new `.bin` files to an output directory.
3. Point OpenPCDet dataset config or symlink to the purified `velodyne` directory.
4. Run `tools/test.py` normally.

This avoids modifying OpenPCDet internals too early.

Later, if offline preprocessing is too slow, implement a dataloader hook.

## 9. Evaluation tables to build

### Table 0: clean detector baseline

- Detector: PointPillars first
- Dataset: KITTI val
- Metric: clean mAP

### Table 1: no-defense ASR

- Detector: PointPillars
- Attacks: Sun20 injection, Wang21 perturbation first
- Metric: ASR and attacked mAP

### Table 2: ASAP defense results

- No defense
- Spherical projection only
- Uniform SPU diffusion
- ASAP full

Metrics:

- clean mAP
- attacked mAP
- ASR
- runtime per frame

### Table 3: ablation

Ablation means removing one component at a time to prove each component matters.

Recommended ablations:

- fixed radius instead of adaptive radius
- no anomaly gate
- no spherical projection
- no batch diffusion

## 10. Immediate next actions

### User-side action

Download KITTI 3D object detection data and place it under:

```text
/root/autodl-tmp/ASAP/data/kitti/
```

Required first:

- Velodyne point clouds
- Camera calibration matrices
- Training labels
- Left color images

Optional later:

- Testing split
- Road planes
- Depth maps only if using CaDDN

### Assistant-side action after data is ready

1. Verify KITTI directory structure and file counts.
2. Generate OpenPCDet KITTI info files.
3. Download or place the matching KITTI PointPillars checkpoint.
4. Run clean PointPillars validation.
5. Record clean mAP in ASAP docs.
6. Start attack baseline implementation or adaptation.

## 11. Key principle

The first engineering goal is not to make ASAP strong immediately. The first goal is to build a trustworthy experimental loop:

```text
KITTI data -> clean detector eval -> attacked eval -> purified eval -> tables
```

Once this loop is stable, ASAP modules can be improved without changing the evaluation foundation.
