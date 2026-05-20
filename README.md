# ASAP

ASAP = Adaptive Spherical Anomaly-Guided Purification.

This repository implements a detector-agnostic LiDAR point cloud adversarial purification framework for 3D object detection.

## Goal

ASAP is designed for LiDAR point cloud attacks, especially point perturbation and point injection attacks. The core idea is to combine:

- Adaptive spherical purification units
- Anomaly-guided SPU selection
- Selective diffusion purification

The proposal document is kept in `docs/01_SA-SPD_proposal.html`.

## Repository layout

```text
ASAP/
├── configs/          # Experiment and method configs
├── docs/             # Proposal, plan and human-readable notes
│   └── references/   # Local reference PDFs and literature index
├── scripts/          # Reproducible command-line entry scripts
├── src/asap/         # ASAP Python package
├── tests/            # Unit tests
├── third_party/      # External repositories or adapters, see policy below
├── data/             # Local datasets, ignored by Git
├── outputs/          # Experiment outputs, ignored by Git
└── weights/          # Checkpoints and pretrained weights, ignored by Git
```

## Project boundary

ASAP is an independent project. All assets used by ASAP must live under `/root/autodl-tmp/ASAP`. Other directories on the same machine, in particular `/root/autodl-tmp/LiDAR_SPD/`, are treated as legacy and are not part of ASAP. ASAP must not import code, configs, weights, or data through paths that point outside this repository, except through the controlled `third_party/` mechanism described below.

## Third-party code policy

When ASAP needs an external project, for example OpenPCDet or a reference attack implementation, the project should be brought in under `third_party/` using these rules:

- **Re-clone fresh from upstream** instead of copying from any pre-existing local copy.
- **Remove the inner `.git` directory** of the cloned repository, so this ASAP repository does not contain another nested Git repository.
- **Preserve the upstream `LICENSE` file** inside the cloned project.
- **Document the upstream URL, commit hash, license, and any modifications** in `third_party/README.md` and, when relevant, in this top-level README under "Acknowledgements".
- **Do not commit large binary or data files** belonging to the upstream project. The directory `third_party/**` is ignored by `.gitignore` by default, except for documentation files. Code files needed for ASAP should be either kept as a local copy outside Git or selectively whitelisted in `.gitignore`.

## Reference literature policy

Reference papers are stored under `docs/references/` for local reading only.

- PDF files of papers are not committed to Git, due to publisher copyright. They are ignored through `.gitignore`.
- A literature index that lists titles, venues, years, and download links is kept in `docs/references/README.md` and committed to Git.
- When summarizing or quoting papers in our own writing, attribution is required.

## KITTI data preparation

ASAP uses the official KITTI 3D Object Detection dataset for the first detection benchmark. KITTI data is not included in this repository and is ignored by Git under `data/`.

The expected local root is:

```text
/root/autodl-tmp/ASAP/data/kitti/
```

For a different machine, set `ASAP_ROOT`, `KITTI_ROOT`, and `DOWNLOAD_DIR` before running the scripts.

### 1. Download and extract KITTI object data

This downloads the four required KITTI 3D Object Detection archives and extracts them under `data/kitti/`:

```bash
bash scripts/download_kitti_object.sh
```

The script downloads:

- `data_object_velodyne.zip`
- `data_object_calib.zip`
- `data_object_label_2.zip`
- `data_object_image_2.zip`

Expected extracted directories:

```text
data/kitti/
├── training/
│   ├── calib/
│   ├── image_2/
│   ├── label_2/
│   └── velodyne/
└── testing/
    ├── calib/
    ├── image_2/
    └── velodyne/
```

### 2. Download KITTI ImageSets

This downloads the OpenPCDet KITTI split files and generates `trainval.txt` from `train.txt + val.txt`:

```bash
bash scripts/download_kitti_imagesets.sh
```

Expected files:

```text
data/kitti/ImageSets/
├── train.txt
├── val.txt
├── trainval.txt
└── test.txt
```

### 3. Verify KITTI object data

This checks the official KITTI object files:

```bash
python scripts/verify_kitti_object.py --root /root/autodl-tmp/ASAP/data/kitti
```

Expected counts:

```text
training/velodyne: 7481
training/calib:    7481
training/label_2:  7481
training/image_2:  7481
testing/velodyne:  7518
testing/calib:     7518
testing/image_2:   7518
```

### 4. Verify KITTI ImageSets

This checks the split files and confirms `trainval = train union val`:

```bash
python scripts/verify_kitti_imagesets.py --root /root/autodl-tmp/ASAP/data/kitti
```

Expected counts:

```text
train.txt:     3712
val.txt:       3769
trainval.txt:  7481
test.txt:      7518
```

### Optional one-command preparation

To run all four steps in sequence:

```bash
bash scripts/prepare_kitti.sh
```

## Current implementation plan

1. Prepare KITTI 3D object detection data and detector baseline.
2. Build no-defense attack baselines.
3. Implement geometric frontend: adaptive radius and SPU builder.
4. Implement spherical projection for injection attack purification.
5. Implement anomaly scorer with compactness, anisotropy, vMF kappa, and density features.
6. Implement selective diffusion interface and batching.
7. Connect ASAP to OpenPCDet-style detection evaluation.

The full plan is kept in `docs/02_implementation_plan.md`.

## Git policy

Large artifacts are intentionally ignored:

- datasets under `data/`
- checkpoints under `weights/` and `checkpoints/`
- logs and experiment outputs under `outputs/`
- nested third-party code trees under `third_party/`
- reference PDFs under `docs/references/`
- Python caches and virtual environments

## Acknowledgements

ASAP builds on ideas from prior work in LiDAR-based 3D detection and adversarial defense. Specific upstream projects that may be vendored under `third_party/` (for example OpenPCDet) keep their original license and copyright. Their license files and upstream attribution are preserved inside `third_party/<project_name>/`.
