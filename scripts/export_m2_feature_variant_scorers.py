#!/usr/bin/env python3
"""Export scorer JSONs for selected E5.2 M2 feature variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from asap.scoring.anomaly_scorer import ScorerConfig, calibrate_scorer


ROOT = Path(__file__).resolve().parents[1]
FEATURES = {
    "f_c": 0,
    "f_a": 1,
    "f_v": 2,
    "f_d": 3,
}
VARIANTS = {
    "full": ("f_c", "f_a", "f_v", "f_d"),
    "only_f_c": ("f_c",),
    "only_f_a": ("f_a",),
    "only_f_v": ("f_v",),
    "only_f_d": ("f_d",),
    "drop_f_c": ("f_a", "f_v", "f_d"),
    "drop_f_a": ("f_c", "f_v", "f_d"),
    "drop_f_v": ("f_c", "f_a", "f_d"),
    "drop_f_d": ("f_c", "f_a", "f_v"),
}


def mask_features(features: np.ndarray, active_names: tuple[str, ...]) -> np.ndarray:
    out = np.zeros_like(features)
    idx = [FEATURES[name] for name in active_names]
    out[:, idx] = features[:, idx]
    return out


def zero_inactive_weights(cfg: ScorerConfig, active_names: tuple[str, ...]) -> ScorerConfig:
    active = set(active_names)
    if "f_c" not in active:
        cfg.w_c = 0.0
    if "f_a" not in active:
        cfg.w_a = 0.0
    if "f_v" not in active:
        cfg.w_v = 0.0
    if "f_d" not in active:
        cfg.w_d = 0.0
    return cfg


def export_variant(attack: str, variant: str, out_dir: Path) -> Path:
    feature_path = ROOT / "outputs" / "ablations" / "E5.2_m2_features" / f"{attack}_features.npz"
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant {variant}. Options: {sorted(VARIANTS)}")

    active = VARIANTS[variant]
    data = np.load(feature_path)
    clean = mask_features(data["clean"], active)
    attacked = mask_features(data["attacked"], active)
    cfg = calibrate_scorer(clean, attacked)
    cfg = zero_inactive_weights(cfg, active)
    cfg.attack_kind = attack

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scorer_{attack}_{variant}.json"
    cfg.to_json(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack", default="perturbation", choices=["injection", "perturbation", "dropping"])
    parser.add_argument("--variant", action="append", required=True, choices=sorted(VARIANTS))
    parser.add_argument(
        "--out_dir",
        default=str(ROOT / "outputs" / "ablations" / "E5.2_m2_features" / "configs"),
    )
    args = parser.parse_args()

    for variant in args.variant:
        path = export_variant(args.attack, variant, Path(args.out_dir))
        print(f"{variant}: {path}")


if __name__ == "__main__":
    main()
