"""Build Track A revised-policy summaries with Dropping skip.

This policy keeps the learned M3 branches that passed the detector-extension
gate and skips purification for E2.3 Dropping:

- E2.1 Injection: score-net + local anchor b075 + Injection radius filter.
- E2.2 Perturbation: score-net + local anchor b075.
- E2.3 Dropping: no M3 / use the attacked no-defense scan.

The Dropping skip is not a new detector run. It is exactly the already parsed
no-defense result for E2.3, because skipping M3 leaves the attacked scan
unchanged.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_PP_BASE = Path("outputs/asap/kitti_pointpillars_eval_summary.json")
DEFAULT_PP_TRACKA = Path(
    "outputs/asap_vpsde_score_net_trackA_policy/"
    "kitti_pointpillars_eval_summary.json"
)
DEFAULT_PVRCNN_TRACKA = Path(
    "outputs/asap_vpsde_score_net_trackA_policy/"
    "kitti_pvrcnn_eval_summary.json"
)
DEFAULT_OUT = Path(
    "outputs/asap_vpsde_score_net_trackA_revised_dropping_skip/"
    "kitti_policy_eval_summary.json"
)


POLICY = {
    "E2.1_injection": {
        "rule": "score-net + local anchor b075 + Injection radius filter",
        "pointpillars_label": "E2.1 ASAP-vpsde-score-net-trackA-policy",
        "pvrcnn_label": "E2.1 TrackA-policy",
        "variant": "b075-ifilter",
        "skip_m3": False,
    },
    "E2.2_perturbation": {
        "rule": "score-net + local anchor b075",
        "pointpillars_label": "E2.2 ASAP-vpsde-score-net-trackA-policy",
        "pvrcnn_label": "E2.2 TrackA-policy",
        "variant": "b075",
        "skip_m3": False,
    },
    "E2.3_dropping": {
        "rule": "skip M3 / no purification",
        "pointpillars_label": "E2.3 no-defense",
        "pvrcnn_label": "E2.3 no-defense",
        "variant": "skip",
        "skip_m3": True,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _with_asr(row: dict[str, Any], clean_car_mod: float) -> dict[str, Any]:
    out = deepcopy(row)
    car_mod = float(out["Car_3d_R40_mod"])
    out["Car_3d_R40_mod_ASR_vs_clean"] = 1.0 - car_mod / clean_car_mod
    return out


def _build_detector_block(
    *,
    detector: str,
    clean: dict[str, Any],
    source: dict[str, Any],
    base_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean_car = float(clean["Car_3d_R40_mod"])
    block: dict[str, Any] = {
        "clean": deepcopy(clean),
        "rows": {},
    }

    for attack_id, spec in POLICY.items():
        label = (
            spec["pointpillars_label"]
            if detector == "PointPillars"
            else spec["pvrcnn_label"]
        )
        if spec["skip_m3"] and base_source is not None:
            src = base_source[label]
        else:
            src = source[label]

        row = _with_asr(src, clean_car)
        row.update(
            {
                "attack_id": attack_id,
                "policy_rule": spec["rule"],
                "variant": spec["variant"],
                "skip_m3": bool(spec["skip_m3"]),
                "source_label": label,
            }
        )
        block["rows"][attack_id] = row
    return block


def build_summary(
    pointpillars_base_path: Path,
    pointpillars_tracka_path: Path,
    pvrcnn_tracka_path: Path,
) -> dict[str, Any]:
    pp_base = _load_json(pointpillars_base_path)
    pp_tracka = _load_json(pointpillars_tracka_path)
    pvrcnn = _load_json(pvrcnn_tracka_path)

    summary: dict[str, Any] = {
        "_metadata": {
            "policy_name": "Track A revised policy with Dropping skip",
            "rationale": (
                "E4.2 showed the learned M3 branch improves Injection and "
                "Perturbation on PV-RCNN but regresses Dropping. Since the "
                "current E2.3 attack removes evidence, the revised policy "
                "skips purification for Dropping and uses the attacked scan "
                "unchanged."
            ),
            "rule": {attack_id: spec["rule"] for attack_id, spec in POLICY.items()},
            "source_summaries": {
                "pointpillars_base": str(pointpillars_base_path),
                "pointpillars_tracka": str(pointpillars_tracka_path),
                "pvrcnn_tracka_and_nodef": str(pvrcnn_tracka_path),
            },
        },
        "PointPillars": _build_detector_block(
            detector="PointPillars",
            clean=pp_base["clean"],
            source=pp_tracka,
            base_source=pp_base,
        ),
        "PV-RCNN": _build_detector_block(
            detector="PV-RCNN",
            clean=pvrcnn["clean"],
            source=pvrcnn,
            base_source=pvrcnn,
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointpillars_base", type=Path, default=DEFAULT_PP_BASE)
    parser.add_argument("--pointpillars_tracka", type=Path, default=DEFAULT_PP_TRACKA)
    parser.add_argument("--pvrcnn_tracka", type=Path, default=DEFAULT_PVRCNN_TRACKA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    summary = build_summary(
        pointpillars_base_path=args.pointpillars_base,
        pointpillars_tracka_path=args.pointpillars_tracka,
        pvrcnn_tracka_path=args.pvrcnn_tracka,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    for detector in ("PointPillars", "PV-RCNN"):
        print(detector)
        for attack_id, row in summary[detector]["rows"].items():
            print(
                f"  {attack_id}: Car_mod={row['Car_3d_R40_mod']:.2f}, "
                f"mAP_mod={row['mAP3D_R40_mod']:.2f}, "
                f"ASR={100.0 * row['Car_3d_R40_mod_ASR_vs_clean']:.1f}%, "
                f"variant={row['variant']}"
            )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
