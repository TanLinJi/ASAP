"""Build the KITTI/PointPillars pre-revision Track A policy summary.

This historical Track A policy is attack-conditional:

- E2.1 Injection: score-net + local anchor b075 + Injection radius filter.
- E2.2 Perturbation: score-net + local anchor b075.
- E2.3 Dropping: score-net + local anchor b075.

This script combines the already parsed OpenPCDet summaries into one JSON file
so paper tables can reference a single artifact.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_B075 = Path(
    "outputs/asap_vpsde_score_net_anchor_b075/kitti_pointpillars_eval_summary.json"
)
DEFAULT_IFILTER = Path(
    "outputs/asap_vpsde_score_net_anchor_b075_ifilter/"
    "kitti_pointpillars_eval_summary.json"
)
DEFAULT_OUT = Path(
    "outputs/asap_vpsde_score_net_trackA_policy/"
    "kitti_pointpillars_eval_summary.json"
)
DEFAULT_SMOKE_GLOB = "outputs/smoke/b075_ifilter_disabled_non_injection_*/summary.json"

POLICY = {
    "E2.1 ASAP-vpsde-score-net-trackA-policy": {
        "summary": "ifilter",
        "source_label": "E2.1 ASAP-vpsde-score-net-b075-ifilter",
        "attack_id": "E2.1_injection",
        "variant": "b075-ifilter",
        "score_anchor_blend": 0.75,
        "score_injection_filter_enabled": True,
    },
    "E2.2 ASAP-vpsde-score-net-trackA-policy": {
        "summary": "b075",
        "source_label": "E2.2 ASAP-vpsde-score-net-b075",
        "attack_id": "E2.2_perturbation",
        "variant": "b075",
        "score_anchor_blend": 0.75,
        "score_injection_filter_enabled": False,
    },
    "E2.3 ASAP-vpsde-score-net-trackA-policy": {
        "summary": "b075",
        "source_label": "E2.3 ASAP-vpsde-score-net-b075",
        "attack_id": "E2.3_dropping",
        "variant": "b075",
        "score_anchor_blend": 0.75,
        "score_injection_filter_enabled": False,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _latest_smoke_summary(glob_pattern: str) -> Path | None:
    matches = sorted(Path(".").glob(glob_pattern))
    if not matches:
        return None
    return matches[-1]


def build_summary(
    b075_path: Path,
    ifilter_path: Path,
    clean_car_mod: float,
    smoke_summary_path: Path | None,
) -> dict[str, Any]:
    summaries = {
        "b075": _load_json(b075_path),
        "ifilter": _load_json(ifilter_path),
    }

    out: dict[str, Any] = {
        "_metadata": {
            "policy_name": "Track A attack-conditional learned M3",
            "clean_reference": {
                "dataset": "KITTI val",
                "detector": "PointPillars",
                "Car_3d_R40_mod": clean_car_mod,
            },
            "rule": {
                "E2.1_injection": "score-net + local anchor b075 + Injection radius filter",
                "E2.2_perturbation": "score-net + local anchor b075",
                "E2.3_dropping": "score-net + local anchor b075",
            },
            "source_summaries": {
                "b075": str(b075_path),
                "ifilter": str(ifilter_path),
            },
        }
    }

    if smoke_summary_path is not None and smoke_summary_path.exists():
        out["_metadata"]["filter_disabled_smoke"] = {
            "path": str(smoke_summary_path),
            "summary": _load_json(smoke_summary_path),
        }

    for target_label, spec in POLICY.items():
        src = summaries[spec["summary"]][spec["source_label"]]
        entry = deepcopy(src)
        car_mod = float(entry["Car_3d_R40_mod"])
        entry.update(
            {
                "source_label": spec["source_label"],
                "attack_id": spec["attack_id"],
                "variant": spec["variant"],
                "score_anchor_blend": spec["score_anchor_blend"],
                "score_injection_filter_enabled": spec[
                    "score_injection_filter_enabled"
                ],
                "Car_3d_R40_mod_ASR_vs_clean": 1.0 - car_mod / clean_car_mod,
            }
        )
        out[target_label] = entry

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b075", type=Path, default=DEFAULT_B075)
    parser.add_argument("--ifilter", type=Path, default=DEFAULT_IFILTER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--clean_car_mod", type=float, default=78.40)
    parser.add_argument(
        "--filter_disabled_smoke",
        type=Path,
        default=None,
        help="Optional non-injection smoke summary JSON. Defaults to latest smoke.",
    )
    args = parser.parse_args()

    smoke = args.filter_disabled_smoke
    if smoke is None:
        smoke = _latest_smoke_summary(DEFAULT_SMOKE_GLOB)

    summary = build_summary(
        b075_path=args.b075,
        ifilter_path=args.ifilter,
        clean_car_mod=args.clean_car_mod,
        smoke_summary_path=smoke,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    for label, row in summary.items():
        if label.startswith("_"):
            continue
        print(
            f"{label}: Car_mod={row['Car_3d_R40_mod']:.2f}, "
            f"mAP_mod={row['mAP3D_R40_mod']:.2f}, "
            f"ASR={100.0 * row['Car_3d_R40_mod_ASR_vs_clean']:.1f}%"
        )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
