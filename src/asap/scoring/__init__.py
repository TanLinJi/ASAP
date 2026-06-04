"""ASAP M2 — 4-feature anomaly scorer for Spherical Purification Units.

Features (paper §3.4):
    f_c : compactness    = |P_in| / V_2          (M2.1)
    f_a : PCA anisotropy = 1 - lambda_3 / sum    (M2.2)
    f_v : vMF concentration kappa-hat            (M2.3)
    f_d : density ratio  = (|P_in|/V_2) / (|P_an|/V_an + eps)  (M2.4)

Aggregation (M2.5):
    s(p) = sigma( w_c * f_c~ + w_a * f_a~ + w_v * f_v~ + w_d * f_d~ + b )
"""

from .anomaly_scorer import (
    AnomalyScorer,
    ScorerConfig,
    compute_features,
    score_spus,
)

__all__ = [
    "AnomalyScorer",
    "ScorerConfig",
    "compute_features",
    "score_spus",
]
