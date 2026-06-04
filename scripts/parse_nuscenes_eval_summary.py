"""Parse OpenPCDet nuScenes metrics_summary.json files.

Example:
    python scripts/parse_nuscenes_eval_summary.py \
        --model cbgs_voxel0075_voxelnext \
        --clean-map 0.605195155366683 \
        --clean-nds 0.6664 \
        --out outputs/attacks/nuscenes_keyframe/nuscenes_voxelnext_no_defense_eval_summary.json \
        --entry "E2.1 no-defense:E4_3_voxelnext_keyframe_no_defense_E2_1"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OPENPCDET_NUSC_BASE = Path(
    "/root/autodl-tmp/ASAP/third_party/OpenPCDet/output/nuscenes_models"
)


def _latest_metrics(model: str, tag: str) -> Path:
    eval_root = OPENPCDET_NUSC_BASE / model / tag / "eval"
    metrics = [
        p
        for p in sorted(eval_root.rglob("metrics_summary.json"))
        if p.stat().st_size > 0
    ]
    if not metrics:
        raise FileNotFoundError(f"No metrics_summary.json under {eval_root}")
    return metrics[-1]


def _load_metrics(path: Path, clean_map: float | None, clean_nds: float | None) -> dict:
    with open(path) as f:
        raw = json.load(f)

    mean_ap = float(raw["mean_ap"])
    nds = float(raw["nd_score"])
    out: dict = {
        "metrics_summary": str(path),
        "mean_ap": mean_ap,
        "nds": nds,
        "mAP_percent": mean_ap * 100.0,
        "NDS_percent": nds * 100.0,
    }

    if clean_map is not None:
        out["clean_mean_ap"] = clean_map
        out["ASR_mAP"] = 1.0 - (mean_ap / clean_map)
        out["ASR_mAP_percent"] = out["ASR_mAP"] * 100.0
    if clean_nds is not None:
        out["clean_nds"] = clean_nds
        out["ASR_NDS"] = 1.0 - (nds / clean_nds)
        out["ASR_NDS_percent"] = out["ASR_NDS"] * 100.0

    class_ap = raw.get("mean_dist_aps", {})
    if class_ap:
        out["class_mAP_percent"] = {
            name: float(value) * 100.0 for name, value in class_ap.items()
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="OpenPCDet nuScenes model output folder, e.g. cbgs_voxel0075_voxelnext.",
    )
    parser.add_argument("--clean-map", type=float, default=None)
    parser.add_argument("--clean-nds", type=float, default=None)
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="If --out already exists, preserve its entries and update/add the requested entries.",
    )
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Format: 'label:tag' (tag = OpenPCDet extra_tag directory).",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    summary: dict = {}
    if args.merge_existing and out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path) as f:
            summary = json.load(f)

    for spec in args.entry:
        label, tag = spec.split(":", 1)
        metrics_path = _latest_metrics(args.model, tag)
        summary[label] = _load_metrics(metrics_path, args.clean_map, args.clean_nds)
        item = summary[label]
        asr = item.get("ASR_mAP_percent", float("nan"))
        print(
            f"[{label}] {tag}  "
            f"mAP={item['mAP_percent']:.2f}  NDS={item['NDS_percent']:.2f}  "
            f"ASR(mAP)={asr:.2f}%"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
