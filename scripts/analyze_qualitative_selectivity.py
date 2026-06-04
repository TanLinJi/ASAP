#!/usr/bin/env python3
"""Generate qualitative selectivity summaries and BEV plots for ASAP.

This script is intentionally read-only with respect to experiment inputs. It
uses existing clean / attacked / defended point clouds and metadata to produce
small paper-facing diagnostic figures for E5.5.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "qualitative" / "E5.5_selectivity"
FIG_DIR = OUT_DIR / "figures"


@dataclass
class Case:
    dataset: str
    attack: str
    frame_id: str
    clean_path: Path | None
    attacked_path: Path
    defended_path: Path
    dim: int
    n_input_points: int
    n_output_points: int
    n_spus: int
    n_flagged_spus: int
    n_score_spus: int
    n_edited_points: int
    n_dropped_points: int
    rank_metric: float

    @property
    def flagged_ratio(self) -> float:
        return self.n_flagged_spus / self.n_spus if self.n_spus else 0.0

    @property
    def score_ratio(self) -> float:
        return self.n_score_spus / self.n_spus if self.n_spus else 0.0

    @property
    def edit_ratio(self) -> float:
        return self.n_edited_points / self.n_input_points if self.n_input_points else 0.0

    @property
    def drop_ratio(self) -> float:
        return self.n_dropped_points / self.n_input_points if self.n_input_points else 0.0


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_points(path: Path, dim: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % dim != 0:
        raise ValueError(f"{path} has {arr.size} floats, not divisible by dim={dim}")
    return arr.reshape(-1, dim)


def top_existing_cases(
    rows: Iterable[dict],
    *,
    dataset: str,
    attack: str,
    dim: int,
    clean_dir: Path | None,
    attacked_dir: Path,
    defended_dir: Path,
    suffix: str,
    metric_key: str,
    limit: int,
) -> list[Case]:
    ranked = []
    for row in rows:
        n_input = int(row.get("n_input_points", 0))
        metric = float(row.get(metric_key, 0)) / n_input if n_input else 0.0
        ranked.append((metric, row))
    ranked.sort(key=lambda x: x[0], reverse=True)

    cases: list[Case] = []
    for metric, row in ranked:
        frame_id = str(row["frame_id"])
        clean_path = None
        if clean_dir is not None:
            clean_path = clean_dir / f"{frame_id}{suffix}"
            if not clean_path.exists():
                clean_path = None
        attacked_path = attacked_dir / f"{frame_id}{suffix}"
        defended_path = defended_dir / f"{frame_id}{suffix}"
        if not attacked_path.exists() or not defended_path.exists():
            continue
        cases.append(
            Case(
                dataset=dataset,
                attack=attack,
                frame_id=frame_id,
                clean_path=clean_path,
                attacked_path=attacked_path,
                defended_path=defended_path,
                dim=dim,
                n_input_points=int(row.get("n_input_points", 0)),
                n_output_points=int(row.get("n_output_points", 0)),
                n_spus=int(row.get("n_spus", 0)),
                n_flagged_spus=int(row.get("n_flagged_spus", 0)),
                n_score_spus=int(row.get("n_score_spus", 0)),
                n_edited_points=int(row.get("n_edited_points", 0)),
                n_dropped_points=int(row.get("n_score_injection_filter_dropped_points", 0)),
                rank_metric=metric,
            )
        )
        if len(cases) >= limit:
            break
    return cases


def axis_limits(*clouds: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    pts = np.concatenate([c[:, :2] for c in clouds if len(c)], axis=0)
    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    pad = np.maximum((hi - lo) * 0.08, 1.0)
    return (float(lo[0] - pad[0]), float(hi[0] + pad[0])), (float(lo[1] - pad[1]), float(hi[1] + pad[1]))


def scatter_bev(ax, pts: np.ndarray, title: str, color: str, max_points: int = 50000) -> None:
    if len(pts) > max_points:
        rng = np.random.default_rng(7)
        pts = pts[rng.choice(len(pts), size=max_points, replace=False)]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.12, c=color, alpha=0.55, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)


def plot_case(case: Case) -> Path:
    attacked = load_points(case.attacked_path, case.dim)
    defended = load_points(case.defended_path, case.dim)
    clean = load_points(case.clean_path, case.dim) if case.clean_path else None

    ncols = 3 if clean is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 4.0), constrained_layout=True)
    if ncols == 2:
        axes = np.asarray(axes)

    if clean is not None:
        scatter_bev(axes[0], clean, "Clean", "#4a5568")
        scatter_bev(axes[1], attacked, "Attacked", "#c2410c")
        scatter_bev(axes[2], defended, "ASAP output", "#047857")
        xlim, ylim = axis_limits(clean, attacked, defended)
    else:
        scatter_bev(axes[0], attacked, "Attacked", "#c2410c")
        scatter_bev(axes[1], defended, "ASAP output", "#047857")
        xlim, ylim = axis_limits(attacked, defended)

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)

    fig.suptitle(
        f"{case.dataset} {case.attack} | {case.frame_id} | "
        f"flagged {case.flagged_ratio:.1%}, edited {case.edit_ratio:.1%}, dropped {case.drop_ratio:.1%}",
        fontsize=10,
    )
    safe_id = case.frame_id.replace("/", "_").replace(":", "_")
    out = FIG_DIR / f"{case.dataset}_{case.attack}_{safe_id}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def write_summary(cases: list[Case], fig_paths: dict[str, Path]) -> None:
    csv_path = OUT_DIR / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "attack",
                "frame_id",
                "n_input_points",
                "n_output_points",
                "flagged_spu_ratio",
                "score_spu_ratio",
                "edited_point_ratio",
                "dropped_point_ratio",
                "figure",
            ],
        )
        writer.writeheader()
        for case in cases:
            key = f"{case.dataset}:{case.frame_id}"
            writer.writerow(
                {
                    "dataset": case.dataset,
                    "attack": case.attack,
                    "frame_id": case.frame_id,
                    "n_input_points": case.n_input_points,
                    "n_output_points": case.n_output_points,
                    "flagged_spu_ratio": f"{case.flagged_ratio:.6f}",
                    "score_spu_ratio": f"{case.score_ratio:.6f}",
                    "edited_point_ratio": f"{case.edit_ratio:.6f}",
                    "dropped_point_ratio": f"{case.drop_ratio:.6f}",
                    "figure": str(fig_paths[key].relative_to(ROOT)),
                }
            )

    md_path = OUT_DIR / "summary.md"
    lines = [
        "# E5.5 qualitative selectivity summary",
        "",
        "| Dataset | Attack | Frame | Flagged SPUs | Score SPUs | Edited points | Dropped points | Figure |",
        "|---------|--------|-------|-------------:|-----------:|--------------:|---------------:|--------|",
    ]
    for case in cases:
        key = f"{case.dataset}:{case.frame_id}"
        fig_rel = fig_paths[key].relative_to(ROOT)
        lines.append(
            f"| {case.dataset} | {case.attack} | `{case.frame_id}` | "
            f"{case.flagged_ratio:.2%} | {case.score_ratio:.2%} | "
            f"{case.edit_ratio:.2%} | {case.drop_ratio:.2%} | `{fig_rel}` |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- KITTI perturbation examples show coordinate editing without global point deletion.",
            "- nuScenes pfilter examples show the disclosed score-net + support-filter cascade; the drop ratio is reported explicitly.",
            "- These figures are qualitative diagnostics, not additional detector metrics.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    kitti_rows = load_jsonl(
        ROOT / "outputs/asap_vpsde_score_net_anchor_b075/kitti/E2.2_perturbation/meta.jsonl"
    )
    kitti_cases = top_existing_cases(
        kitti_rows,
        dataset="kitti",
        attack="perturbation",
        dim=4,
        clean_dir=ROOT / "data/kitti/training/velodyne",
        attacked_dir=ROOT / "outputs/attacks/kitti/E2.2_perturbation/velodyne_attacked",
        defended_dir=ROOT / "outputs/asap_vpsde_score_net_anchor_b075/kitti/E2.2_perturbation/velodyne_purified",
        suffix=".bin",
        metric_key="n_edited_points",
        limit=3,
    )

    nusc_rows = load_jsonl(
        ROOT / "outputs/defenses/nuscenes_keyframe_accepted_asap_pfilter/asap/E2.2_perturbation_eps0p5/meta.jsonl"
    )
    nusc_cases = top_existing_cases(
        nusc_rows,
        dataset="nuscenes",
        attack="perturbation_pfilter",
        dim=5,
        clean_dir=ROOT / "data/nuscenes/v1.0-trainval/samples/LIDAR_TOP",
        attacked_dir=ROOT / "outputs/attacks/nuscenes_keyframe_budget_gate/E2.2_perturbation_eps0p5/samples/LIDAR_TOP",
        defended_dir=ROOT / "outputs/defenses/nuscenes_keyframe_accepted_asap_pfilter/asap/E2.2_perturbation_eps0p5/samples/LIDAR_TOP",
        suffix=".bin",
        metric_key="n_score_injection_filter_dropped_points",
        limit=3,
    )

    cases = kitti_cases + nusc_cases
    if not cases:
        raise RuntimeError("No qualitative cases found")

    fig_paths: dict[str, Path] = {}
    for case in cases:
        fig_paths[f"{case.dataset}:{case.frame_id}"] = plot_case(case)
    write_summary(cases, fig_paths)

    print(f"Wrote {len(cases)} qualitative cases to {OUT_DIR}")


if __name__ == "__main__":
    main()
