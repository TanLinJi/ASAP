"""ASAP geometry module — Adaptive-Radius Spherical Purification Unit (SPU)
construction (M1).

Exports:
    AdaptiveSPUBuilder: end-to-end builder that turns an (N, 3+) point
        cloud into a list of SPU dicts with adaptive outer / inner radii.
    SPU: dataclass-like dict structure (see build_spus for keys).
"""

from .adaptive_spu import (
    AdaptiveSPUBuilder,
    SPUConfig,
    build_spus,
    build_spus_torch,
    farthest_point_sampling,
    knn_distances,
)

__all__ = [
    "AdaptiveSPUBuilder",
    "SPUConfig",
    "build_spus",
    "build_spus_torch",
    "farthest_point_sampling",
    "knn_distances",
]
