"""Parse OpenPCDet KITTI PointPillars eval logs into a summary JSON.

For each (tag) in the run, locates the latest log_eval_*.txt under
`third_party/OpenPCDet/output/kitti_models/<model>/<tag>/eval/`,
extracts the Car/Pedestrian/Cyclist 3D AP_R40 (easy/mod/hard),
average predicted-object count, and recall_0.7, and writes a single
JSON file with one entry per (label).

Usage:
    python scripts/parse_eval_summary.py \
        --out outputs/asap_hybrid/kitti_pointpillars_eval_summary.json \
        --entry "clean:E1.1_pp" \
        --entry "E2.1 ASAP-hybrid:E4_asap_hybrid_E2.1_pp" \
        --entry "E2.2 ASAP-hybrid:E4_asap_hybrid_E2.2_pp" \
        --entry "E2.3 ASAP-hybrid:E4_asap_hybrid_E2.3_pp"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

OPENPCDET_BASE = Path(
    "/root/autodl-tmp/ASAP/third_party/OpenPCDet/output/kitti_models"
)


def _latest_log(model: str, tag: str) -> Path:
    eval_root = OPENPCDET_BASE / model / tag / "eval"
    logs = [p for p in sorted(eval_root.rglob("log_eval_*.txt")) if p.stat().st_size > 0]
    if not logs:
        raise FileNotFoundError(f"No non-empty log_eval_*.txt under {eval_root}")
    return logs[-1]


_CLASS_BLOCK = {
    "Car": ("Car AP_R40@0.70, 0.70, 0.70:", "Car_3d_R40"),
    "Pedestrian": ("Pedestrian AP_R40@0.50, 0.50, 0.50:", "Pedestrian_3d_R40"),
    "Cyclist": ("Cyclist AP_R40@0.50, 0.50, 0.50:", "Cyclist_3d_R40"),
}


def _parse_log(path: Path) -> dict:
    text = path.read_text()
    out: dict = {"log": path.name}

    m = re.search(r"recall_roi_0\.7:\s*([0-9.]+)", text)
    if m and float(m.group(1)) > 0:
        out["recall_0.7"] = float(m.group(1))
    else:
        m = re.search(r"recall_rcnn_0\.7:\s*([0-9.]+)", text)
        if m:
            out["recall_0.7"] = float(m.group(1))

    m = re.search(r"Average predicted number of objects\([0-9]+ samples\):\s*([0-9.]+)", text)
    if m:
        out["pred_obj"] = float(m.group(1))

    for cls, (header, prefix) in _CLASS_BLOCK.items():
        idx = text.find(header)
        if idx < 0:
            continue
        block = text[idx : idx + 600]
        m = re.search(r"3d\s+AP:\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)", block)
        if not m:
            continue
        easy, mod, hard = (float(x) for x in m.groups())
        out[f"{prefix}_easy"] = easy
        out[f"{prefix}_mod"] = mod
        out[f"{prefix}_hard"] = hard

    mods = [out[k] for k in ("Car_3d_R40_mod", "Pedestrian_3d_R40_mod", "Cyclist_3d_R40_mod") if k in out]
    if len(mods) == 3:
        out["mAP3D_R40_mod"] = sum(mods) / 3.0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model",
        default="pointpillar",
        help="OpenPCDet KITTI model output folder (default: pointpillar).",
    )
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Format: 'label:tag' (tag = OpenPCDet extra_tag directory).",
    )
    args = parser.parse_args()

    summary: dict = {}
    for spec in args.entry:
        label, tag = spec.split(":", 1)
        log = _latest_log(args.model, tag)
        summary[label] = _parse_log(log)
        car_mod = summary[label].get("Car_3d_R40_mod", float("nan"))
        m3d = summary[label].get("mAP3D_R40_mod", float("nan"))
        print(f"[{label}] {tag}  Car_mod={car_mod:.2f}  mAP_mod={m3d:.2f}  ({log.name})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
