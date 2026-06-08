"""End-to-end ASAP pipeline: M1 -> M2 -> M3.

Given an input point cloud P and (optionally) a pre-calibrated
ScorerConfig + PurifierConfig, return the purified cloud P_hat plus a
metadata dict describing what was edited.

Provides a CLI that:
  1) Loads a calibration JSON (or fits a fresh one on a small clean
     vs. attacked subset).
  2) Iterates over all .bin files in --input_dir and writes purified
     .bin files to --out_dir, mirroring the KITTI naming convention.
  3) Logs per-frame edit statistics to a JSONL metadata file.

Calibration JSON schema (see asap.scoring.ScorerConfig.to_json).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from asap.geometry import AdaptiveSPUBuilder, SPUConfig
from asap.scoring import AnomalyScorer, ScorerConfig, compute_features, score_spus
from asap.scoring.anomaly_scorer import calibrate_scorer
from asap.diffusion import DropPurifier, HybridPurifier, VPSDEPurifier, PurifierConfig

logger = logging.getLogger(__name__)


def _now() -> float:
    return time.perf_counter()


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ASAPPipeline:
    """End-to-end M1 -> M2 -> M3 pipeline."""

    def __init__(
        self,
        spu_cfg: Optional[SPUConfig] = None,
        scorer_cfg: Optional[ScorerConfig] = None,
        purifier_cfg: Optional[PurifierConfig] = None,
        purifier_method: str = "drop",
    ):
        self.spu_cfg = spu_cfg or SPUConfig()
        self.scorer_cfg = scorer_cfg or ScorerConfig()
        self.purifier_cfg = purifier_cfg or PurifierConfig(tau=self.scorer_cfg.tau)
        # Keep tau consistent with calibrated scorer tau.
        self.purifier_cfg.tau = self.scorer_cfg.tau

        self.builder = AdaptiveSPUBuilder(self.spu_cfg)
        self.scorer = AnomalyScorer(self.scorer_cfg)
        if purifier_method == "drop":
            self.purifier = DropPurifier(self.purifier_cfg)
        elif purifier_method == "vp_sde":
            self.purifier = VPSDEPurifier(self.purifier_cfg)
        elif purifier_method == "hybrid":
            self.purifier = HybridPurifier(self.purifier_cfg)
        else:
            raise ValueError(f"Unknown purifier_method: {purifier_method}")
        self.purifier_method = purifier_method

    def run(
        self,
        points: np.ndarray,
        seed: int = 0,
        profile: bool = False,
    ) -> Tuple[np.ndarray, dict]:
        """Purify a single point cloud.

        Args:
            points: (N, C) input cloud.
            seed: RNG seed for the SPU builder (FPS).

        Returns:
            (purified_points, meta_dict)
        """
        t_total = _now()

        t = _now()
        spus = self.builder.build(points, seed=seed)
        m1_ms = _elapsed_ms(t)

        t = _now()
        features, scores = self.scorer.score(points, spus)
        m2_ms = _elapsed_ms(t)

        t = _now()
        purified, p_meta = self.purifier.purify(points, spus, scores)
        m3_ms = _elapsed_ms(t)

        meta = {
            "purifier_method": self.purifier_method,
            "tau": self.scorer_cfg.tau,
            "n_input_points": int(len(points)),
            "n_output_points": int(len(purified)),
            "n_spus": len(spus),
            "n_flagged_spus": int((scores > self.scorer_cfg.tau).sum()),
            **p_meta,
        }
        if profile:
            meta["profile_m1_ms"] = round(m1_ms, 3)
            meta["profile_m2_ms"] = round(m2_ms, 3)
            meta["profile_m3_ms"] = round(m3_ms, 3)
            meta["profile_total_compute_ms"] = round(_elapsed_ms(t_total), 3)
        return purified, meta


# ---------------------------------------------------------------------------
# Calibration helper (clean dir + attacked dir -> ScorerConfig)
# ---------------------------------------------------------------------------

def calibrate_from_dirs(
    clean_dir: str,
    attacked_dir: str,
    n_frames: int = 20,
    spu_cfg: Optional[SPUConfig] = None,
    seed: int = 42,
    drop_density_thresh: float = 0.5,
    num_features: int = 4,
) -> ScorerConfig:
    """Fit a ScorerConfig from paired clean / attacked .bin directories.

    Generic attack-type detection (handles add / replace / drop):
      For every SPU centered at a clean point, count inner-ball points
      in BOTH the clean and the attacked cloud. The SPU is marked
      "attacked" if either:
        (a) any inner point in the *attacked* cloud is absent from the
            clean cloud  (added or perturbed-to-new-position), OR
        (b) the inner-ball point count drops by more than
            `drop_density_thresh` (e.g. half the points disappeared).
      The features fed to the logistic regressor come from the
      *attacked* cloud (which is what M2 sees at inference time).

    Args:
        clean_dir: directory of clean .bin files.
        attacked_dir: directory of attacked .bin files (same filenames).
        n_frames: number of paired frames to use for calibration.
        spu_cfg: SPUConfig; defaults if None.
        seed: RNG seed.
        drop_density_thresh: fraction of inner-ball points that must
            have disappeared to mark an SPU as "attacked".

    Returns:
        Fitted ScorerConfig.
    """
    from scipy.spatial import cKDTree

    clean_dir = Path(clean_dir)
    attacked_dir = Path(attacked_dir)
    files = sorted(attacked_dir.glob("*.bin"))[:n_frames]
    if not files:
        raise FileNotFoundError(f"No .bin files in {attacked_dir}")

    builder = AdaptiveSPUBuilder(spu_cfg or SPUConfig())

    all_clean, all_atk = [], []
    n_added_total = 0
    n_dropped_total = 0
    n_clean_pts_total = 0
    for f in files:
        clean_path = clean_dir / f.name
        if not clean_path.exists():
            continue
        pc = np.fromfile(clean_path, dtype=np.float32).reshape(-1, num_features)
        pa = np.fromfile(f, dtype=np.float32).reshape(-1, num_features)

        # Membership: which points in attacked cloud are new (added or perturbed)?
        # Which clean points were dropped (or perturbed away)?
        # Quantised-coord hashing at 1mm.
        clean_keys = set(map(tuple, np.round(pc[:, :3] * 1000).astype(np.int64)))
        atk_keys = np.round(pa[:, :3] * 1000).astype(np.int64)
        atk_keys_set = set(map(tuple, atk_keys))
        added_in_atk_mask = np.array(
            [tuple(k) not in clean_keys for k in atk_keys], dtype=bool
        )
        n_added_total += int(added_in_atk_mask.sum())
        # Clean points whose 1mm-key is missing from the attacked cloud.
        clean_keys_arr = np.round(pc[:, :3] * 1000).astype(np.int64)
        n_dropped_total += int(sum(
            tuple(k) not in atk_keys_set for k in clean_keys_arr
        ))
        n_clean_pts_total += len(pc)

        sc = builder.build(pc, seed=seed)
        sa = builder.build(pa, seed=seed)
        feats_clean = compute_features(pc, sc)
        feats_atk = compute_features(pa, sa)
        all_clean.append(feats_clean)

        # Strategy A (add / perturb): SPUs in *attacked* cloud whose inner
        # ball touches an added/moved point.
        spu_is_attacked_a = np.zeros(len(sa), dtype=bool)
        for i, s in enumerate(sa):
            spu_is_attacked_a[i] = bool(added_in_atk_mask[s["inner_idx"]].any())

        # Strategy B (drop): SPUs in *clean* cloud whose inner-ball
        # density dropped >= drop_density_thresh after the attack.
        # We need to compare the same physical region, so for each clean SPU
        # query the attacked-cloud tree at r=r2 and check the count.
        atk_tree = cKDTree(pa[:, :3])
        spu_is_dropped_b = np.zeros(len(sc), dtype=bool)
        for i, s in enumerate(sc):
            n_clean_inner = len(s["inner_idx"])
            if n_clean_inner == 0:
                continue
            center = pc[s["center_idx"], :3]
            r2 = s["r2"]
            n_atk_inner = len(atk_tree.query_ball_point(center, r=r2))
            if (n_clean_inner - n_atk_inner) / max(n_clean_inner, 1) >= drop_density_thresh:
                spu_is_dropped_b[i] = True

        # Combine: features from attacked cloud where add/perturb detected,
        # plus features from the corresponding clean-SPU region in the
        # attacked cloud where drop detected. For the dropped SPUs we
        # re-extract features at the *same physical center* in the attacked
        # cloud so the scorer sees what M2 would see at inference time.
        if spu_is_attacked_a.sum() > 0:
            all_atk.append(feats_atk[spu_is_attacked_a])

        if spu_is_dropped_b.sum() > 0:
            # Build SPUs *for the dropped centers* on the attacked cloud,
            # using the same centers (so we can compute features post-drop).
            dropped_centers = [int(sc[i]["center_idx"]) for i in np.where(spu_is_dropped_b)[0]]
            # Convert clean-cloud center indices to attacked-cloud indices:
            # find nearest atk point to each clean center.
            clean_centers_xyz = pc[dropped_centers, :3]
            _, atk_center_idx = atk_tree.query(clean_centers_xyz, k=1)
            # Manually construct minimal SPU dicts to reuse compute_features.
            dropped_spus = []
            for j, ci in enumerate(atk_center_idx):
                # Use same r1, r2 as the original clean SPU
                src = sc[dropped_centers.index(dropped_centers[j])] if False else \
                      sc[np.where(spu_is_dropped_b)[0][j]]
                center = pa[ci, :3]
                sphere = atk_tree.query_ball_point(center, r=src["r1"])
                if len(sphere) < 4:
                    continue
                sphere_arr = np.asarray(sphere, dtype=np.int64)
                d2 = np.einsum("ij,ij->i", pa[sphere_arr, :3] - center, pa[sphere_arr, :3] - center)
                inner_mask = d2 <= src["r2"] ** 2
                dropped_spus.append({
                    "center_idx": int(ci),
                    "center": center.astype(np.float32),
                    "r1": src["r1"],
                    "r2": src["r2"],
                    "sphere_idx": sphere_arr,
                    "inner_idx": sphere_arr[inner_mask],
                    "annulus_idx": sphere_arr[~inner_mask],
                })
            if dropped_spus:
                all_atk.append(compute_features(pa, dropped_spus))

    if not all_clean or not all_atk:
        raise RuntimeError("Calibration produced no clean or attacked SPUs.")

    clean_feats = np.concatenate(all_clean)
    atk_feats = np.concatenate(all_atk)
    logger.info(
        f"Calibration: {len(clean_feats)} clean SPUs, {len(atk_feats)} attacked SPUs"
    )
    cfg = calibrate_scorer(clean_feats, atk_feats)

    # Infer attack_kind from added vs dropped point ratio.
    #
    # Heuristic (relative to clean cloud size):
    #   - "injection":     n_added >> n_dropped, both relative to total points.
    #     Attack inserts new points; few/no clean points disappear.
    #   - "dropping":      n_dropped >> n_added.
    #     Attack removes existing points; few/no new points appear.
    #   - "perturbation":  n_added ≈ n_dropped, both substantial.
    #     Each moved point shows up as one drop (old loc) + one add (new loc).
    #
    # The thresholds below are calibrated on the KITTI E2.1/E2.2/E2.3 sets;
    # they are intentionally generous so users can override with a dedicated
    # CLI flag if a future attack falls between buckets.
    eps = 1e-9
    add_frac = n_added_total / max(n_clean_pts_total, 1)
    drop_frac = n_dropped_total / max(n_clean_pts_total, 1)
    ratio = n_added_total / max(n_dropped_total, eps)
    if ratio >= 5.0 and drop_frac < 0.02:
        kind = "injection"
    elif ratio <= 0.2 and add_frac < 0.02:
        kind = "dropping"
    elif add_frac >= 0.05 and drop_frac >= 0.05 and 0.5 <= ratio <= 2.0:
        kind = "perturbation"
    else:
        # Fall back to whichever side dominates.
        if add_frac > drop_frac:
            kind = "injection"
        elif drop_frac > add_frac:
            kind = "dropping"
        else:
            kind = "perturbation"
    cfg.attack_kind = kind
    logger.info(
        f"Calibration attack-prior: kind={kind} "
        f"(added={n_added_total}, dropped={n_dropped_total}, "
        f"add_frac={add_frac:.4f}, drop_frac={drop_frac:.4f}, ratio={ratio:.2f})"
    )
    return cfg


# ---------------------------------------------------------------------------
# Batch runner over a directory of .bin files
# ---------------------------------------------------------------------------

# Module-level pickleable worker (cannot be a closure for multiprocessing).
_GLOBAL_PIPELINE_KW = None


def _worker_init(spu_cfg, scorer_cfg, purifier_cfg, purifier_method):
    global _GLOBAL_PIPELINE_KW
    _GLOBAL_PIPELINE_KW = ASAPPipeline(
        spu_cfg=spu_cfg,
        scorer_cfg=scorer_cfg,
        purifier_cfg=purifier_cfg,
        purifier_method=purifier_method,
    )


def _worker_run(args):
    bf_str, out_str, seed, num_features, profile = args
    t_total = _now()
    t = _now()
    pts = np.fromfile(bf_str, dtype=np.float32).reshape(-1, num_features)
    io_read_ms = _elapsed_ms(t)
    out_pts, meta = _GLOBAL_PIPELINE_KW.run(pts, seed=seed, profile=profile)
    t = _now()
    out_pts.astype(np.float32).tofile(out_str)
    io_write_ms = _elapsed_ms(t)
    if profile:
        meta["profile_io_read_ms"] = round(io_read_ms, 3)
        meta["profile_io_write_ms"] = round(io_write_ms, 3)
        meta["profile_total_ms"] = round(_elapsed_ms(t_total), 3)
    meta["frame_id"] = Path(bf_str).stem
    return meta


def run_pipeline_on_dir(
    pipeline: ASAPPipeline,
    input_dir: str,
    out_dir: str,
    meta_jsonl: Optional[str] = None,
    seed: int = 0,
    workers: int = 1,
    num_features: int = 4,
    profile: bool = False,
) -> None:
    """Apply the pipeline to every .bin in input_dir; write purified .bin
    files and one JSONL metadata record per frame.

    If workers > 1, frames are processed in parallel via multiprocessing.
    Each worker re-instantiates ASAPPipeline once at init time.
    """
    in_path = Path(input_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    meta_fh = open(meta_jsonl, "w") if meta_jsonl else None

    bin_files = sorted(in_path.glob("*.bin"))
    if not bin_files:
        logger.error(f"No .bin files in {in_path}")
        return

    t0 = time.time()
    total_in, total_out = 0, 0

    if workers <= 1:
        for idx, bf in enumerate(bin_files):
            t_frame = _now()
            t = _now()
            pts = np.fromfile(str(bf), dtype=np.float32).reshape(-1, num_features)
            io_read_ms = _elapsed_ms(t)
            out_pts, meta = pipeline.run(pts, seed=seed, profile=profile)
            total_in += len(pts)
            total_out += len(out_pts)
            t = _now()
            out_pts.astype(np.float32).tofile(str(out_path / bf.name))
            io_write_ms = _elapsed_ms(t)
            if meta_fh:
                meta["frame_id"] = bf.stem
                if profile:
                    meta["profile_io_read_ms"] = round(io_read_ms, 3)
                    meta["profile_io_write_ms"] = round(io_write_ms, 3)
                    meta["profile_total_ms"] = round(_elapsed_ms(t_frame), 3)
                meta_fh.write(json.dumps(meta) + "\n")

            if (idx + 1) % 200 == 0 or idx == len(bin_files) - 1:
                elapsed = time.time() - t0
                avg = elapsed / (idx + 1)
                eta = avg * (len(bin_files) - idx - 1)
                logger.info(
                    f"[asap-pipeline] {idx+1}/{len(bin_files)} frames, "
                    f"in={total_in} out={total_out} "
                    f"({100*(total_in-total_out)/max(total_in,1):.1f}% edited), "
                    f"avg={avg*1000:.0f}ms/frame, eta={eta:.0f}s"
                )
    else:
        import multiprocessing as mp
        # Build (in_path, out_path, seed) tuples for every frame.
        tasks = [
            (str(bf), str(out_path / bf.name), seed, num_features, profile)
            for bf in bin_files
        ]
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(
                pipeline.spu_cfg,
                pipeline.scorer_cfg,
                pipeline.purifier_cfg,
                pipeline.purifier_method,
            ),
        ) as pool:
            for idx, meta in enumerate(pool.imap_unordered(_worker_run, tasks, chunksize=4)):
                total_in += meta.get("n_input_points", 0)
                total_out += meta.get("n_output_points", 0)
                if meta_fh:
                    meta_fh.write(json.dumps(meta) + "\n")
                if (idx + 1) % 200 == 0 or idx == len(bin_files) - 1:
                    elapsed = time.time() - t0
                    avg = elapsed / (idx + 1)
                    eta = avg * (len(bin_files) - idx - 1)
                    logger.info(
                        f"[asap-pipeline] {idx+1}/{len(bin_files)} frames, "
                        f"in={total_in} out={total_out} "
                        f"({100*(total_in-total_out)/max(total_in,1):.1f}% edited), "
                        f"avg={avg*1000:.0f}ms/frame, eta={eta:.0f}s, workers={workers}"
                    )

    if meta_fh:
        meta_fh.close()
    logger.info(f"ASAP pipeline done: {len(bin_files)} frames -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="ASAP end-to-end purification pipeline")
    parser.add_argument("--input_dir", required=True, help="Dir of attacked .bin files")
    parser.add_argument("--out_dir", required=True, help="Output dir for purified .bin")
    parser.add_argument("--meta_jsonl", default=None, help="Per-frame metadata JSONL")
    parser.add_argument("--purifier", choices=["drop", "vp_sde", "hybrid"], default="drop")
    parser.add_argument("--tau", type=float, default=None, help="Override decision threshold")
    parser.add_argument(
        "--policy_min_votes",
        type=int,
        default=None,
        help=(
            "Minimum number of flagged SPU inner-ball votes required before a point "
            "is edited or dropped by the purifier. Default uses the purifier config."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_features",
        type=int,
        default=4,
        help="Point feature dimension in each .bin file: KITTI=4, nuScenes=5.",
    )
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default 1).")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Add per-frame timing breakdown to metadata JSONL.",
    )

    # M1 SPU construction knobs. Defaults preserve the paper's ASAP setting.
    parser.add_argument("--spu_k", type=int, default=16, help="M1 k-NN local scale k.")
    parser.add_argument("--spu_alpha", type=float, default=2.0, help="M1 radius multiplier alpha.")
    parser.add_argument("--spu_beta", type=float, default=0.67, help="M1 inner-radius multiplier beta.")
    parser.add_argument("--spu_r_min", type=float, default=0.10, help="M1 minimum outer radius in meters.")
    parser.add_argument("--spu_r_max", type=float, default=0.40, help="M1 maximum outer radius in meters.")
    parser.add_argument("--spu_eta", type=float, default=0.30, help="M1 FPS stride in meters.")
    parser.add_argument("--spu_max_centers", type=int, default=None, help="Optional cap on FPS centers.")
    parser.add_argument("--spu_min_points_per_spu", type=int, default=4, help="Drop SPUs smaller than this.")
    parser.add_argument(
        "--spu_backend",
        choices=["cpu", "torch"],
        default="cpu",
        help="M1 backend: cpu=cKDTree, torch=experimental torch distance kernels.",
    )
    parser.add_argument("--spu_torch_device", default="cuda", help="Device for --spu_backend torch.")
    parser.add_argument(
        "--spu_torch_chunk_centers",
        type=int,
        default=512,
        help="Center chunk size for --spu_backend torch radius/kNN queries.",
    )

    # Calibration: either load an existing JSON or fit one on the fly.
    cal = parser.add_mutually_exclusive_group(required=True)
    cal.add_argument("--scorer_json", help="Pre-calibrated ScorerConfig JSON")
    cal.add_argument(
        "--calibrate_with_clean_dir",
        help="Clean .bin dir, paired with --input_dir, used to fit a scorer",
    )
    parser.add_argument(
        "--calibrate_n_frames", type=int, default=20,
        help="Number of frame pairs to use when fitting a scorer (default 20).",
    )
    parser.add_argument(
        "--save_scorer_json", default=None,
        help="If set, write fitted ScorerConfig JSON here.",
    )
    parser.add_argument(
        "--calibrate_only",
        action="store_true",
        help="Fit/load the scorer config and exit before applying purification.",
    )

    # Hybrid M3 knobs (only used when --purifier hybrid).
    parser.add_argument(
        "--score_net_ckpt",
        default=None,
        help="Track-A VP-SDE score-net checkpoint for --purifier vp_sde.",
    )
    parser.add_argument(
        "--score_net_device",
        default="cuda",
        help="Device used to load --score_net_ckpt (default cuda).",
    )
    parser.add_argument(
        "--score_step_size",
        type=float,
        default=1.0,
        help="Step size for score-net denoising update in normalized SPU coordinates.",
    )
    parser.add_argument(
        "--score_clip",
        type=float,
        default=3.0,
        help="Clamp normalized score-net coordinates to [-score_clip, score_clip].",
    )
    parser.add_argument(
        "--score_batch_spus",
        type=int,
        default=64,
        help=(
            "Batch score-net inference across SPUs with identical inner point counts. "
            "Set to 1 for the old per-SPU forward path."
        ),
    )
    parser.add_argument(
        "--score_anchor_blend",
        type=float,
        default=0.0,
        help=(
            "Blend score-net output toward the local-mean anchor for edited points. "
            "0 = pure score-net, 1 = local-mean anchor."
        ),
    )
    parser.add_argument(
        "--score_injection_filter_radius",
        type=float,
        default=0.0,
        help=(
            "Optional Injection-only radius filter after score-net denoising. "
            "Set >0 with --score_injection_filter_min_neighbors >0 to enable."
        ),
    )
    parser.add_argument(
        "--score_injection_filter_min_neighbors",
        type=int,
        default=0,
        help="Minimum neighbors for the Injection-only post score-net radius filter.",
    )
    parser.add_argument(
        "--score_filter_attack_kinds",
        default="injection",
        help=(
            "Comma-separated attack kinds that enable the post score-net radius filter. "
            "Default keeps historical behavior: injection only."
        ),
    )
    parser.add_argument(
        "--t_star",
        type=float,
        default=None,
        help="Override VP-SDE truncation time for --purifier vp_sde.",
    )

    # Hybrid M3 knobs (only used when --purifier hybrid).
    parser.add_argument(
        "--hybrid_anisotropy_threshold", type=float, default=None,
        help="Override HybridPurifier cluster-drop anisotropy threshold.",
    )
    parser.add_argument(
        "--hybrid_score_multiplier", type=float, default=None,
        help="Override HybridPurifier absolute score gate (multiplier on tau).",
    )
    parser.add_argument(
        "--hybrid_drop_score_quantile", type=float, default=None,
        help="Override HybridPurifier dynamic score quantile gate.",
    )
    parser.add_argument(
        "--hybrid_isolation_radius", type=float, default=None,
        help="Override HybridPurifier isolation-radius drop radius (m). Set 0 to disable branch.",
    )
    parser.add_argument(
        "--hybrid_min_neighbors", type=int, default=None,
        help="Override HybridPurifier isolation-drop min-neighbors. Set 0 to disable branch.",
    )
    parser.add_argument(
        "--attack_kind",
        choices=["injection", "perturbation", "dropping"],
        default=None,
        help="Override attack-kind prior used by attack-conditional purifiers.",
    )

    args = parser.parse_args()

    # Build / load scorer config
    if args.scorer_json:
        scorer_cfg = ScorerConfig.from_json(args.scorer_json)
        logger.info(f"Loaded ScorerConfig from {args.scorer_json}")
    else:
        logger.info(f"Calibrating scorer on first {args.calibrate_n_frames} paired frames...")
        scorer_cfg = calibrate_from_dirs(
            clean_dir=args.calibrate_with_clean_dir,
            attacked_dir=args.input_dir,
            n_frames=args.calibrate_n_frames,
            seed=args.seed,
            num_features=args.num_features,
        )
        if args.save_scorer_json:
            scorer_cfg.to_json(args.save_scorer_json)
            logger.info(f"Saved fitted ScorerConfig -> {args.save_scorer_json}")

    if args.tau is not None:
        scorer_cfg.tau = args.tau
        logger.info(f"Overriding tau = {args.tau}")

    if args.calibrate_only:
        if args.scorer_json:
            logger.info("--calibrate_only with --scorer_json: loaded scorer config; no output written.")
        elif not args.save_scorer_json:
            logger.warning("--calibrate_only used without --save_scorer_json; fitted scorer was not saved.")
        logger.info("Calibration-only mode complete.")
        return

    purifier_cfg = PurifierConfig(tau=scorer_cfg.tau)
    if args.policy_min_votes is not None:
        purifier_cfg.policy_min_votes = max(1, int(args.policy_min_votes))
    if args.t_star is not None:
        purifier_cfg.t_star = args.t_star
    if args.score_net_ckpt is not None:
        purifier_cfg.score_net_ckpt = args.score_net_ckpt
        purifier_cfg.score_net_device = args.score_net_device
        purifier_cfg.score_step_size = args.score_step_size
        purifier_cfg.score_clip = args.score_clip
        purifier_cfg.score_batch_spus = args.score_batch_spus
        purifier_cfg.score_anchor_blend = args.score_anchor_blend
        purifier_cfg.score_injection_filter_radius = args.score_injection_filter_radius
        purifier_cfg.score_injection_filter_min_neighbors = (
            args.score_injection_filter_min_neighbors
        )
        purifier_cfg.score_filter_attack_kinds = args.score_filter_attack_kinds
    if scorer_cfg.attack_kind is not None:
        purifier_cfg.attack_kind = scorer_cfg.attack_kind
    if args.attack_kind is not None:
        purifier_cfg.attack_kind = args.attack_kind
    if args.hybrid_anisotropy_threshold is not None:
        purifier_cfg.hybrid_anisotropy_threshold = args.hybrid_anisotropy_threshold
    if args.hybrid_score_multiplier is not None:
        purifier_cfg.hybrid_score_multiplier = args.hybrid_score_multiplier
    if args.hybrid_drop_score_quantile is not None:
        purifier_cfg.hybrid_drop_score_quantile = args.hybrid_drop_score_quantile
    if args.hybrid_isolation_radius is not None:
        purifier_cfg.hybrid_isolation_radius = args.hybrid_isolation_radius
    if args.hybrid_min_neighbors is not None:
        purifier_cfg.hybrid_min_neighbors = args.hybrid_min_neighbors

    spu_cfg = SPUConfig(
        k=args.spu_k,
        alpha=args.spu_alpha,
        beta=args.spu_beta,
        r_min=args.spu_r_min,
        r_max=args.spu_r_max,
        eta=args.spu_eta,
        max_centers=args.spu_max_centers,
        min_points_per_spu=args.spu_min_points_per_spu,
        backend=args.spu_backend,
        torch_device=args.spu_torch_device,
        torch_chunk_centers=args.spu_torch_chunk_centers,
    )

    pipeline = ASAPPipeline(
        spu_cfg=spu_cfg,
        scorer_cfg=scorer_cfg,
        purifier_cfg=purifier_cfg,
        purifier_method=args.purifier,
    )

    run_pipeline_on_dir(
        pipeline,
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        meta_jsonl=args.meta_jsonl,
        seed=args.seed,
        workers=args.workers,
        num_features=args.num_features,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
