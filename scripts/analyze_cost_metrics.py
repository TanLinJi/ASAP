#!/usr/bin/env python3
"""Summarize ASAP selective-ratio and preprocessing-cost evidence.

This script consumes per-frame ``meta.jsonl`` files emitted by
``python -m asap.pipeline`` and optional pipeline logs containing
``avg=<ms>ms/frame`` progress lines. It writes CSV/JSON/Markdown summaries
under ``outputs/cost`` for E6.1/E6.2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as stats
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    setting: str
    policy: str
    meta: Path
    logs: tuple[Path, ...] = ()
    note: str = ""


DEFAULT_RUNS = [
    RunSpec(
        "KITTI",
        "PointPillars / Injection",
        "TrackA b075-ifilter",
        Path("outputs/asap_vpsde_score_net_anchor_b075_ifilter/kitti/E2.1_injection/meta.jsonl"),
        (
            Path("outputs/asap_vpsde_score_net_anchor_b075_ifilter/logs/asap_vpsde_score_net_E2.1_injection_gpu0.log"),
            Path("outputs/asap_vpsde_score_net_anchor_b075_ifilter/logs/asap_vpsde_score_net_E2.1_injection_gpu1.log"),
        ),
        "Final KITTI Injection policy.",
    ),
    RunSpec(
        "KITTI",
        "PointPillars / Perturbation",
        "TrackA b075",
        Path("outputs/asap_vpsde_score_net_anchor_b075/kitti/E2.2_perturbation/meta.jsonl"),
        (
            Path("outputs/asap_vpsde_score_net_anchor_b075/logs/asap_vpsde_score_net_E2.2_perturbation_gpu0.log"),
            Path("outputs/asap_vpsde_score_net_anchor_b075/logs/asap_vpsde_score_net_E2.2_perturbation_gpu1.log"),
        ),
        "Final KITTI Perturbation policy.",
    ),
    RunSpec(
        "KITTI",
        "PointPillars / Dropping",
        "Score-net diagnostic",
        Path("outputs/asap_vpsde_score_net_anchor_b075/kitti/E2.3_dropping/meta.jsonl"),
        (
            Path("outputs/asap_vpsde_score_net_anchor_b075/logs/asap_vpsde_score_net_E2.3_dropping_gpu0.log"),
            Path("outputs/asap_vpsde_score_net_anchor_b075/logs/asap_vpsde_score_net_E2.3_dropping_gpu1.log"),
        ),
        "Diagnostic only; final paper policy skips M3 for Dropping.",
    ),
    RunSpec(
        "nuScenes",
        "VoxelNeXt / Injection",
        "ASAP-v1",
        Path("outputs/defenses/nuscenes_keyframe_accepted/asap/E2.1_injection/meta.jsonl"),
        (
            Path("outputs/defenses/nuscenes_keyframe_accepted/logs/asap_E2.1_injection_gpu0.log"),
            Path("outputs/defenses/nuscenes_keyframe_accepted/logs/asap_E2.1_injection_gpu1.log"),
        ),
        "Same purified output evaluated on VoxelNeXt and TransFusion-Lidar.",
    ),
    RunSpec(
        "nuScenes",
        "VoxelNeXt / Perturbation eps0.5",
        "ASAP-v1",
        Path("outputs/defenses/nuscenes_keyframe_accepted/asap/E2.2_perturbation_eps0p5/meta.jsonl"),
        (
            Path("outputs/defenses/nuscenes_keyframe_accepted/logs/asap_E2.2_perturbation_eps0p5_gpu0.log"),
            Path("outputs/defenses/nuscenes_keyframe_accepted/logs/asap_E2.2_perturbation_eps0p5_gpu1.log"),
        ),
        "Pure score-net policy, below ROR on nuScenes Perturbation.",
    ),
    RunSpec(
        "nuScenes",
        "VoxelNeXt / Perturbation eps0.5",
        "ASAP-pfilter",
        Path("outputs/defenses/nuscenes_keyframe_accepted_asap_pfilter/asap/E2.2_perturbation_eps0p5/meta.jsonl"),
        (
            Path("outputs/defenses/nuscenes_keyframe_accepted_asap_pfilter/logs/asap_E2.2_perturbation_eps0p5_gpu0.log"),
            Path("outputs/defenses/nuscenes_keyframe_accepted_asap_pfilter/logs/asap_E2.2_perturbation_eps0p5_gpu1.log"),
        ),
        "Score-net + support-filter cascade used for final nuScenes Perturbation rows.",
    ),
]


def rel(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return stats.fmean(vals) if vals else float("nan")


def parse_log_ms(paths: Iterable[Path]) -> tuple[float, float, float]:
    """Return mean per-shard ms/frame, aggregate throughput ms/frame, max shard ms."""
    shard_ms: list[float] = []
    pattern = re.compile(r"avg=([0-9.]+)ms/frame")
    for path in paths:
        path = rel(path)
        if not path.exists():
            continue
        last = None
        for match in pattern.finditer(path.read_text(encoding="utf-8", errors="ignore")):
            last = float(match.group(1))
        if last is not None:
            shard_ms.append(last)

    if not shard_ms:
        return float("nan"), float("nan"), float("nan")
    per_shard = mean(shard_ms)
    aggregate = per_shard / len(shard_ms)
    return per_shard, aggregate, max(shard_ms)


def summarize(spec: RunSpec) -> dict:
    meta_path = rel(spec.meta)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    rows = read_jsonl(meta_path)
    if not rows:
        raise ValueError(f"empty meta file: {meta_path}")

    n_spus = [float(r.get("n_spus", 0)) for r in rows]
    n_flagged = [float(r.get("n_flagged_spus", 0)) for r in rows]
    n_score_spus = [float(r.get("n_score_spus", 0)) for r in rows]
    n_input = [float(r.get("n_input_points", 0)) for r in rows]
    n_output = [float(r.get("n_output_points", r.get("n_kept_points", 0))) for r in rows]
    n_edited = [float(r.get("n_edited_points", r.get("n_score_edited_points", 0))) for r in rows]
    n_filter_drop = [float(r.get("n_score_injection_filter_dropped_points", 0)) for r in rows]

    flagged_ratio = [safe_ratio(a, b) for a, b in zip(n_flagged, n_spus)]
    score_ratio = [safe_ratio(a, b) for a, b in zip(n_score_spus, n_spus)]
    edited_ratio = [safe_ratio(a, b) for a, b in zip(n_edited, n_input)]
    output_drop_ratio = [1.0 - safe_ratio(a, b) for a, b in zip(n_output, n_input)]
    filter_drop_ratio = [safe_ratio(a, b) for a, b in zip(n_filter_drop, n_input)]
    per_shard_ms, aggregate_ms, max_shard_ms = parse_log_ms(spec.logs)

    return {
        "dataset": spec.dataset,
        "setting": spec.setting,
        "policy": spec.policy,
        "frames": len(rows),
        "meta_path": str(spec.meta),
        "mean_spus": mean(n_spus),
        "mean_flagged_spus": mean(n_flagged),
        "mean_flagged_ratio": mean(flagged_ratio),
        "p05_flagged_ratio": pct(flagged_ratio, 0.05),
        "p95_flagged_ratio": pct(flagged_ratio, 0.95),
        "mean_score_spu_ratio": mean(score_ratio),
        "mean_edited_point_ratio": mean(edited_ratio),
        "mean_output_drop_ratio": mean(output_drop_ratio),
        "mean_filter_drop_ratio": mean(filter_drop_ratio),
        "total_input_points": int(sum(n_input)),
        "total_output_points": int(sum(n_output)),
        "total_filter_dropped_points": int(sum(n_filter_drop)),
        "per_shard_ms_per_frame": per_shard_ms,
        "aggregate_ms_per_frame_two_gpu": aggregate_ms,
        "max_shard_ms_per_frame": max_shard_ms,
        "note": spec.note,
    }


def fmt_pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{100.0 * x:.2f}%"


def fmt_num(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.1f}"


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# E6 Cost and Selective-Ratio Summary",
        "",
        "This summary is generated from existing ASAP per-frame metadata and pipeline logs.",
        "",
        "| Dataset | Setting | Policy | Frames | Flagged SPUs | Score SPUs | Edited points | Output drop | Filter drop | ms/frame/shard | ms/frame aggregate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {dataset} | {setting} | {policy} | {frames} | {flagged} | {score} | "
            "{edited} | {outdrop} | {fdrop} | {shard} | {agg} |".format(
                dataset=r["dataset"],
                setting=r["setting"],
                policy=r["policy"],
                frames=r["frames"],
                flagged=fmt_pct(r["mean_flagged_ratio"]),
                score=fmt_pct(r["mean_score_spu_ratio"]),
                edited=fmt_pct(r["mean_edited_point_ratio"]),
                outdrop=fmt_pct(r["mean_output_drop_ratio"]),
                fdrop=fmt_pct(r["mean_filter_drop_ratio"]),
                shard=fmt_num(r["per_shard_ms_per_frame"]),
                agg=fmt_num(r["aggregate_ms_per_frame_two_gpu"]),
            )
        )

    lines.extend(
        [
            "",
            "Interpretation notes:",
            "- `Flagged SPUs` is $|C^\\star|/|C|$ from M2 thresholding.",
            "- `Score SPUs` is the subset actually sent through the score-net path when that metadata is available.",
            "- `Output drop` includes any support-filter point removals; it is zero for pure coordinate-editing runs.",
            "- `ms/frame aggregate` divides the mean per-shard latency by the number of GPU shards represented in the logs, so it is a throughput estimate rather than single-frame latency.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/cost/E6_summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = rel(Path(args.out_dir))
    rows = [summarize(spec) for spec in DEFAULT_RUNS]
    write_csv(rows, out_dir / "cost_selective_summary.csv")
    write_json(rows, out_dir / "cost_selective_summary.json")
    write_markdown(rows, out_dir / "cost_selective_summary.md")
    print(f"Wrote {len(rows)} run summaries to {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
