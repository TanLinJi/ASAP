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
├── docs/             # Proposal and human-readable notes
├── scripts/          # Reproducible command-line entry scripts
├── src/asap/         # ASAP Python package
├── tests/            # Unit tests
├── third_party/      # External repositories or adapters
├── data/             # Local datasets, ignored by Git
├── outputs/          # Experiment outputs, ignored by Git
└── weights/          # Checkpoints and pretrained weights, ignored by Git
```

## Current implementation plan

1. Implement geometric frontend: adaptive radius and SPU builder.
2. Implement spherical projection for injection attack purification.
3. Implement anomaly scorer with compactness, anisotropy, vMF kappa, and density features.
4. Implement selective diffusion interface and batching.
5. Connect ASAP to OpenPCDet-style detection evaluation.

## Git policy

Large artifacts are intentionally ignored:

- datasets under `data/`
- checkpoints under `weights/`
- logs and experiment outputs under `outputs/`
- Python caches and virtual environments
