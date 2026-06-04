#!/usr/bin/env python3
"""Lightweight M2 feature-ablation diagnostics.

This script does not run detector inference. It re-extracts the same SPU-level
features used during scorer calibration, then evaluates whether each M2 feature
or leave-one-out feature set separates clean vs attacked SPUs.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from asap.geometry import AdaptiveSPUBuilder, SPUConfig
from asap.scoring import compute_features, score_spus
from asap.scoring.anomaly_scorer import ScorerConfig, calibrate_scorer


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "ablations" / "E5.2_m2_features"
FEATURE_NAMES = ("f_c", "f_a", "f_v", "f_d")


@dataclass
class FeatureSet:
    name: str
    active: tuple[int, ...]


FEATURE_SETS = [
    FeatureSet("full", (0, 1, 2, 3)),
    FeatureSet("only_f_c", (0,)),
    FeatureSet("only_f_a", (1,)),
    FeatureSet("only_f_v", (2,)),
    FeatureSet("only_f_d", (3,)),
    FeatureSet("drop_f_c", (1, 2, 3)),
    FeatureSet("drop_f_a", (0, 2, 3)),
    FeatureSet("drop_f_v", (0, 1, 3)),
    FeatureSet("drop_f_d", (0, 1, 2)),
]


@dataclass
class AttackSpec:
    attack_id: str
    attack_name: str
    clean_dir: Path
    attacked_dir: Path
    num_features: int
    existing_scorer: Path


KITTI_CLEAN = ROOT / "data" / "kitti" / "training" / "velodyne"
ATTACKS = [
    AttackSpec(
        attack_id="E2.1",
        attack_name="injection",
        clean_dir=KITTI_CLEAN,
        attacked_dir=ROOT / "outputs" / "attacks" / "kitti" / "E2.1_injection" / "velodyne_attacked",
        num_features=4,
        existing_scorer=ROOT / "outputs" / "asap" / "configs" / "scorer_E2.1_injection.json",
    ),
    AttackSpec(
        attack_id="E2.2",
        attack_name="perturbation",
        clean_dir=KITTI_CLEAN,
        attacked_dir=ROOT / "outputs" / "attacks" / "kitti" / "E2.2_perturbation" / "velodyne_attacked",
        num_features=4,
        existing_scorer=ROOT / "outputs" / "asap" / "configs" / "scorer_E2.2_perturbation.json",
    ),
    AttackSpec(
        attack_id="E2.3",
        attack_name="dropping",
        clean_dir=KITTI_CLEAN,
        attacked_dir=ROOT / "outputs" / "attacks" / "kitti" / "E2.3_dropping" / "velodyne_attacked",
        num_features=4,
        existing_scorer=ROOT / "outputs" / "asap" / "configs" / "scorer_E2.3_dropping.json",
    ),
]


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC via average ranks, with tie handling."""
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = ranks[labels].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def best_youden(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    """Return best threshold, J, TPR, FPR."""
    labels = labels.astype(bool)
    thresholds = np.unique(scores)
    if len(thresholds) > 2000:
        thresholds = np.quantile(scores, np.linspace(0.0, 1.0, 2001))
        thresholds = np.unique(thresholds)

    best = (0.5, -1.0, 0.0, 1.0)
    for t in thresholds:
        pred = scores > t
        tp = int((pred & labels).sum())
        fn = int((~pred & labels).sum())
        fp = int((pred & ~labels).sum())
        tn = int((~pred & ~labels).sum())
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j_val = tpr - fpr
        if j_val > best[1]:
            best = (float(t), float(j_val), float(tpr), float(fpr))
    return best


def mask_features(features: np.ndarray, active: Iterable[int]) -> np.ndarray:
    out = np.zeros_like(features)
    idx = list(active)
    out[:, idx] = features[:, idx]
    return out


def collect_calibration_features(
    spec: AttackSpec,
    n_frames: int,
    seed: int,
    drop_density_thresh: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Mirror `pipeline.calibrate_from_dirs`, but return feature arrays."""
    builder = AdaptiveSPUBuilder(SPUConfig())
    files = sorted(spec.attacked_dir.glob("*.bin"))[:n_frames]
    if not files:
        raise FileNotFoundError(f"No .bin files in {spec.attacked_dir}")

    all_clean: list[np.ndarray] = []
    all_attacked: list[np.ndarray] = []
    n_added_total = 0
    n_dropped_total = 0
    n_clean_pts_total = 0
    frames_used = 0

    for attacked_path in files:
        clean_path = spec.clean_dir / attacked_path.name
        if not clean_path.exists():
            continue
        pc = np.fromfile(clean_path, dtype=np.float32).reshape(-1, spec.num_features)
        pa = np.fromfile(attacked_path, dtype=np.float32).reshape(-1, spec.num_features)

        clean_keys = set(map(tuple, np.round(pc[:, :3] * 1000).astype(np.int64)))
        atk_keys = np.round(pa[:, :3] * 1000).astype(np.int64)
        atk_keys_set = set(map(tuple, atk_keys))
        added_in_atk_mask = np.array([tuple(k) not in clean_keys for k in atk_keys], dtype=bool)
        n_added_total += int(added_in_atk_mask.sum())
        clean_keys_arr = np.round(pc[:, :3] * 1000).astype(np.int64)
        n_dropped_total += int(sum(tuple(k) not in atk_keys_set for k in clean_keys_arr))
        n_clean_pts_total += len(pc)

        sc = builder.build(pc, seed=seed)
        sa = builder.build(pa, seed=seed)
        feats_clean = compute_features(pc, sc)
        feats_attacked_cloud = compute_features(pa, sa)
        all_clean.append(feats_clean)
        frames_used += 1

        spu_is_attacked_a = np.zeros(len(sa), dtype=bool)
        for i, spu in enumerate(sa):
            spu_is_attacked_a[i] = bool(added_in_atk_mask[spu["inner_idx"]].any())
        if spu_is_attacked_a.sum() > 0:
            all_attacked.append(feats_attacked_cloud[spu_is_attacked_a])

        atk_tree = cKDTree(pa[:, :3])
        spu_is_dropped_b = np.zeros(len(sc), dtype=bool)
        for i, spu in enumerate(sc):
            n_clean_inner = len(spu["inner_idx"])
            if n_clean_inner == 0:
                continue
            center = pc[spu["center_idx"], :3]
            n_atk_inner = len(atk_tree.query_ball_point(center, r=spu["r2"]))
            if (n_clean_inner - n_atk_inner) / max(n_clean_inner, 1) >= drop_density_thresh:
                spu_is_dropped_b[i] = True

        dropped_ids = np.where(spu_is_dropped_b)[0]
        if len(dropped_ids) > 0:
            clean_centers_xyz = pc[[sc[i]["center_idx"] for i in dropped_ids], :3]
            _, atk_center_idx = atk_tree.query(clean_centers_xyz, k=1)
            dropped_spus = []
            for src_idx, ci in zip(dropped_ids, atk_center_idx):
                src = sc[int(src_idx)]
                center = pa[int(ci), :3]
                sphere = atk_tree.query_ball_point(center, r=src["r1"])
                if len(sphere) < 4:
                    continue
                sphere_arr = np.asarray(sphere, dtype=np.int64)
                d = pa[sphere_arr, :3] - center
                inner_mask = np.einsum("ij,ij->i", d, d) <= src["r2"] ** 2
                dropped_spus.append(
                    {
                        "center_idx": int(ci),
                        "center": center.astype(np.float32),
                        "r1": src["r1"],
                        "r2": src["r2"],
                        "sphere_idx": sphere_arr,
                        "inner_idx": sphere_arr[inner_mask],
                        "annulus_idx": sphere_arr[~inner_mask],
                    }
                )
            if dropped_spus:
                all_attacked.append(compute_features(pa, dropped_spus))

    if not all_clean or not all_attacked:
        raise RuntimeError(f"{spec.attack_name}: no calibration features collected")

    meta = {
        "frames_used": frames_used,
        "n_added_total": n_added_total,
        "n_dropped_total": n_dropped_total,
        "n_clean_pts_total": n_clean_pts_total,
        "add_frac": n_added_total / max(n_clean_pts_total, 1),
        "drop_frac": n_dropped_total / max(n_clean_pts_total, 1),
    }
    return np.concatenate(all_clean), np.concatenate(all_attacked), meta


def evaluate_feature_sets(
    clean_feats: np.ndarray,
    attacked_feats: np.ndarray,
    existing_cfg: ScorerConfig,
) -> list[dict]:
    labels = np.concatenate([np.zeros(len(clean_feats)), np.ones(len(attacked_feats))]).astype(bool)
    rows: list[dict] = []
    for feature_set in FEATURE_SETS:
        clean_masked = mask_features(clean_feats, feature_set.active)
        attacked_masked = mask_features(attacked_feats, feature_set.active)
        cfg = calibrate_scorer(clean_masked, attacked_masked)
        scores = score_spus(np.concatenate([clean_masked, attacked_masked]), cfg)
        best_tau, best_j, best_tpr, best_fpr = best_youden(labels, scores)
        rows.append(
            {
                "variant": feature_set.name,
                "active_features": ",".join(FEATURE_NAMES[i] for i in feature_set.active),
                "n_active": len(feature_set.active),
                "auc": auc_rank(labels, scores),
                "youden_j": best_j,
                "best_tau": best_tau,
                "tpr": best_tpr,
                "fpr": best_fpr,
                "cfg_tau": cfg.tau,
                "weights": [cfg.w_c, cfg.w_a, cfg.w_v, cfg.w_d],
            }
        )

    # Existing full scorer on the same extracted features, for sanity.
    scores_existing = score_spus(np.concatenate([clean_feats, attacked_feats]), existing_cfg)
    best_tau, best_j, best_tpr, best_fpr = best_youden(labels, scores_existing)
    rows.append(
        {
            "variant": "existing_full_scorer",
            "active_features": "f_c,f_a,f_v,f_d",
            "n_active": 4,
            "auc": auc_rank(labels, scores_existing),
            "youden_j": best_j,
            "best_tau": best_tau,
            "tpr": best_tpr,
            "fpr": best_fpr,
            "cfg_tau": existing_cfg.tau,
            "weights": [existing_cfg.w_c, existing_cfg.w_a, existing_cfg.w_v, existing_cfg.w_d],
        }
    )
    return rows


def write_outputs(results: list[dict], feature_meta: dict[str, dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "feature_ablation_summary.csv"
    fieldnames = [
        "attack_id",
        "attack_name",
        "variant",
        "active_features",
        "n_clean_spus",
        "n_attacked_spus",
        "auc",
        "youden_j",
        "best_tau",
        "tpr",
        "fpr",
        "cfg_tau",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})

    json_path = OUT_DIR / "feature_ablation_summary.json"
    json_path.write_text(
        json.dumps({"feature_meta": feature_meta, "results": results}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# E5.2 M2 feature ablation diagnostic",
        "",
        "This is a lightweight calibration-set diagnostic. It does not run detector inference.",
        "",
    ]
    for attack in ATTACKS:
        attack_rows = [r for r in results if r["attack_name"] == attack.attack_name]
        attack_rows = sorted(
            [r for r in attack_rows if r["variant"] != "existing_full_scorer"],
            key=lambda r: (r["auc"], r["youden_j"]),
            reverse=True,
        )
        lines.extend(
            [
                f"## {attack.attack_id} {attack.attack_name}",
                "",
                "| Variant | Active features | AUC | Youden-J | TPR | FPR | tau |",
                "|---------|-----------------|----:|---------:|----:|----:|----:|",
            ]
        )
        for row in attack_rows:
            lines.append(
                f"| `{row['variant']}` | `{row['active_features']}` | "
                f"{row['auc']:.4f} | {row['youden_j']:.4f} | "
                f"{row['tpr']:.4f} | {row['fpr']:.4f} | {row['best_tau']:.4f} |"
            )
        lines.append("")

    (OUT_DIR / "feature_ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    feature_meta: dict[str, dict] = {}
    for spec in ATTACKS:
        print(f"[E5.2] collecting features for {spec.attack_id} {spec.attack_name}")
        clean_feats, attacked_feats, meta = collect_calibration_features(
            spec,
            n_frames=20,
            seed=42,
            drop_density_thresh=0.5,
        )
        feature_meta[spec.attack_name] = {
            **meta,
            "n_clean_spus": int(len(clean_feats)),
            "n_attacked_spus": int(len(attacked_feats)),
        }
        np.savez_compressed(
            OUT_DIR / f"{spec.attack_name}_features.npz",
            clean=clean_feats,
            attacked=attacked_feats,
            meta=np.array([json.dumps(feature_meta[spec.attack_name])]),
        )
        existing_cfg = ScorerConfig.from_json(spec.existing_scorer)
        rows = evaluate_feature_sets(clean_feats, attacked_feats, existing_cfg)
        for row in rows:
            row["attack_id"] = spec.attack_id
            row["attack_name"] = spec.attack_name
            row["n_clean_spus"] = int(len(clean_feats))
            row["n_attacked_spus"] = int(len(attacked_feats))
            row["weights"] = [float(v) for v in row["weights"]]
            for key in ["auc", "youden_j", "best_tau", "tpr", "fpr", "cfg_tau"]:
                row[key] = float(row[key])
        results.extend(rows)

    write_outputs(results, feature_meta)
    print(f"[E5.2] wrote diagnostics to {OUT_DIR}")


if __name__ == "__main__":
    main()
