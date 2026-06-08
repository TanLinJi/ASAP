#!/usr/bin/env python
"""Train the first ASAP local VP-SDE score network on clean KITTI SPU patches.

This is deliberately a compact Track-A starter model. It trains on clean
local patches sampled from M1 SPUs and learns the VP-SDE denoising score.

Example smoke run:
    PYTHONPATH=src /root/miniconda3/envs/asap/bin/python scripts/train_vpsde_score_net.py \
        --clean_dir data/kitti/training/velodyne \
        --out_ckpt checkpoints/kitti/asap_score_net_smoke.pth \
        --max_frames 1 --patches_per_frame 8 --epochs 1 --batch_size 4 --device cpu

Dual-GPU nuScenes starter:
    CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src /root/miniconda3/envs/asap/bin/python scripts/train_vpsde_score_net.py \
        --clean_dir data/nuscenes/v1.0-trainval/samples/LIDAR_TOP \
        --out_ckpt checkpoints/nuscenes/asap_score_net_trackA_v1.pth \
        --num_features 5 --data_parallel
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from asap.diffusion.score_net import (
    ScoreNetConfig,
    create_score_net,
    load_score_net_checkpoint,
    save_score_net_checkpoint,
    sample_fixed_patch,
)
from asap.geometry import AdaptiveSPUBuilder, SPUConfig

logger = logging.getLogger(__name__)


LOSS_PROFILES = ("score_mse", "time_sigma2", "geo", "density", "geo_density_pair")


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required; run inside the `asap` conda env.") from exc
    return torch


def load_patch_pool(
    clean_dir: str,
    cfg: ScoreNetConfig,
    max_frames: int,
    patches_per_frame: int,
    seed: int,
    num_features: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    files = sorted(Path(clean_dir).glob("*.bin"))
    if max_frames > 0:
        files = files[:max_frames]
    if not files:
        raise FileNotFoundError(f"No .bin files found in {clean_dir}")

    builder = AdaptiveSPUBuilder(SPUConfig())
    patches = []
    for frame_idx, path in enumerate(files):
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, num_features)
        spus = builder.build(pts, seed=seed + frame_idx)
        if not spus:
            continue
        chosen = rng.choice(len(spus), size=min(patches_per_frame, len(spus)), replace=False)
        xyz = pts[:, :3]
        for si in chosen:
            spu = spus[int(si)]
            inner = xyz[spu["inner_idx"]]
            patch = sample_fixed_patch(
                inner,
                center=spu["center"],
                radius=spu["r2"],
                patch_size=cfg.patch_size,
                rng=rng,
            )
            patch = np.clip(patch, -cfg.max_coord, cfg.max_coord)
            patches.append(patch)
        logger.info("loaded frame %s (%d/%d), patches=%d", path.name, frame_idx + 1, len(files), len(patches))

    if not patches:
        raise RuntimeError("No patches sampled; check clean_dir and SPU settings.")
    return np.stack(patches, axis=0).astype(np.float32)


def sample_paired_fixed_patch(
    clean_xyz: np.ndarray,
    attacked_xyz: np.ndarray,
    center: np.ndarray,
    radius: float,
    patch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample aligned clean/attacked normalized patches for perturbation pairs."""
    radius = float(max(radius, 1e-4))
    if len(clean_xyz) == 0:
        clean_norm = np.zeros((1, 3), dtype=np.float32)
        attacked_norm = np.zeros((1, 3), dtype=np.float32)
    else:
        clean_norm = (clean_xyz - center[None, :]) / radius
        attacked_norm = (attacked_xyz - center[None, :]) / radius
    replace = len(clean_norm) < patch_size
    idx = rng.choice(len(clean_norm), size=patch_size, replace=replace)
    return clean_norm[idx].astype(np.float32), attacked_norm[idx].astype(np.float32)


def load_paired_patch_pool(
    clean_dir: str,
    attacked_dir: str,
    cfg: ScoreNetConfig,
    max_frames: int,
    patches_per_frame: int,
    seed: int,
    num_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load aligned clean/attacked patch pairs for E2.2 perturbation fine-tuning."""
    rng = np.random.default_rng(seed)
    clean_root = Path(clean_dir)
    attacked_root = Path(attacked_dir)
    files = [p for p in sorted(attacked_root.glob("*.bin")) if (clean_root / p.name).exists()]
    if max_frames > 0:
        files = files[:max_frames]
    if not files:
        raise FileNotFoundError(f"No paired .bin files found in {clean_dir} and {attacked_dir}")

    builder = AdaptiveSPUBuilder(SPUConfig())
    clean_patches = []
    attacked_patches = []
    for frame_idx, attacked_path in enumerate(files):
        clean_path = clean_root / attacked_path.name
        clean_pts = np.fromfile(clean_path, dtype=np.float32).reshape(-1, num_features)
        attacked_pts = np.fromfile(attacked_path, dtype=np.float32).reshape(-1, num_features)
        if len(clean_pts) != len(attacked_pts):
            logger.warning(
                "skipping %s: clean/attacked point counts differ (%d vs %d)",
                attacked_path.name,
                len(clean_pts),
                len(attacked_pts),
            )
            continue
        spus = builder.build(clean_pts, seed=seed + frame_idx)
        if not spus:
            continue
        chosen = rng.choice(len(spus), size=min(patches_per_frame, len(spus)), replace=False)
        clean_xyz = clean_pts[:, :3]
        attacked_xyz = attacked_pts[:, :3]
        for si in chosen:
            spu = spus[int(si)]
            idx = np.asarray(spu["inner_idx"], dtype=np.int64)
            clean_patch, attacked_patch = sample_paired_fixed_patch(
                clean_xyz[idx],
                attacked_xyz[idx],
                center=spu["center"],
                radius=spu["r2"],
                patch_size=cfg.patch_size,
                rng=rng,
            )
            clean_patches.append(np.clip(clean_patch, -cfg.max_coord, cfg.max_coord))
            attacked_patches.append(np.clip(attacked_patch, -cfg.max_coord, cfg.max_coord))
        logger.info(
            "loaded paired frame %s (%d/%d), pair_patches=%d",
            attacked_path.name,
            frame_idx + 1,
            len(files),
            len(clean_patches),
        )

    if not clean_patches:
        raise RuntimeError("No paired patches sampled; check clean_dir and attacked_dir.")
    return (
        np.stack(clean_patches, axis=0).astype(np.float32),
        np.stack(attacked_patches, axis=0).astype(np.float32),
    )


def score_matching_loss(pred, target, sigma, profile: str):
    """Compute a configurable denoising-score matching loss.

    `score_mse` is the original Track A objective. `time_sigma2`, `geo`,
    and `density` multiply each sample by sigma_t^2 so tiny-noise timesteps
    do not dominate updates. Auxiliary terms are added outside this function.
    """
    err = (pred - target) ** 2
    per_sample = err.mean(dim=(1, 2))
    if profile == "score_mse":
        weight = 1.0
    elif profile in {"time_sigma2", "geo", "density", "geo_density_pair"}:
        weight = sigma.detach() ** 2
    else:
        raise ValueError(f"Unknown loss profile: {profile}")
    return (per_sample * weight).mean()


def patch_covariance(x):
    """Return per-patch 3D covariance matrices for [B, N, 3] patches."""
    centered = x - x.mean(dim=1, keepdim=True)
    denom = max(int(x.shape[1]) - 1, 1)
    return centered.transpose(1, 2).matmul(centered) / denom


def geometry_consistency_loss(x_hat, x0, lambda_chamfer: float, lambda_centroid: float, lambda_cov: float):
    """Preserve local patch shape after the one-step Tweedie estimate."""
    zero = x0.new_zeros(())

    if lambda_chamfer:
        torch = require_torch()
        dist = torch.cdist(x_hat.float(), x0.float())
        chamfer = dist.min(dim=2).values.mean(dim=1) + dist.min(dim=1).values.mean(dim=1)
        chamfer = chamfer.mean()
    else:
        chamfer = zero

    if lambda_centroid:
        centroid = ((x_hat.mean(dim=1) - x0.mean(dim=1)) ** 2).mean()
    else:
        centroid = zero

    if lambda_cov:
        cov = ((patch_covariance(x_hat) - patch_covariance(x0)) ** 2).mean()
    else:
        cov = zero

    total = lambda_chamfer * chamfer + lambda_centroid * centroid + lambda_cov * cov
    return total, {
        "chamfer": chamfer.detach(),
        "centroid": centroid.detach(),
        "cov": cov.detach(),
        "geo": total.detach(),
    }


def density_spacing_loss(x_hat, x0, density_k: int):
    """Preserve local kNN spacing without pointwise reconstruction pressure."""
    torch = require_torch()
    n = int(x0.shape[1])
    if n <= 1:
        return x0.new_zeros(())
    k = max(1, min(int(density_k), n - 1))

    def mean_knn_distance(x):
        dist = torch.cdist(x.float(), x.float())
        eye = torch.eye(n, device=x.device, dtype=torch.bool)[None, :, :]
        dist = dist.masked_fill(eye, float("inf"))
        return dist.topk(k, largest=False, dim=2).values.mean(dim=2)

    return (mean_knn_distance(x_hat) - mean_knn_distance(x0)).abs().mean()


def train(args):
    torch = require_torch()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = ScoreNetConfig(
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        time_dim=args.time_dim,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        t_eps=args.t_eps,
        t_max=args.t_max,
    )
    patches = load_patch_pool(
        clean_dir=args.clean_dir,
        cfg=cfg,
        max_frames=args.max_frames,
        patches_per_frame=args.patches_per_frame,
        seed=args.seed,
        num_features=args.num_features,
    )
    logger.info("training patches: %s", patches.shape)

    pair_clean_np = None
    pair_attacked_np = None
    pair_clean = None
    pair_attacked = None
    if args.loss_profile == "geo_density_pair" or args.pair_fraction > 0.0:
        if not args.paired_attacked_dir:
            raise ValueError("--paired_attacked_dir is required for paired fine-tuning")
        pair_clean_np, pair_attacked_np = load_paired_patch_pool(
            clean_dir=args.clean_dir,
            attacked_dir=args.paired_attacked_dir,
            cfg=cfg,
            max_frames=args.paired_max_frames if args.paired_max_frames is not None else args.max_frames,
            patches_per_frame=args.paired_patches_per_frame,
            seed=args.seed + 10000,
            num_features=args.num_features,
        )
        logger.info("paired patches: clean=%s attacked=%s", pair_clean_np.shape, pair_attacked_np.shape)

    device = torch.device(args.device)
    if args.init_ckpt:
        model, ckpt_cfg, _ = load_score_net_checkpoint(args.init_ckpt, device=args.device)
        if ckpt_cfg != cfg:
            raise ValueError(f"--init_ckpt config {ckpt_cfg} does not match requested config {cfg}")
        logger.info("initialized model from %s", args.init_ckpt)
    else:
        model = create_score_net(cfg).to(device)
    use_data_parallel = bool(args.data_parallel)
    if use_data_parallel:
        if device.type != "cuda":
            logger.warning("--data_parallel requested on non-CUDA device; disabling it.")
            use_data_parallel = False
        elif torch.cuda.device_count() < 2:
            logger.warning(
                "--data_parallel requested but only %d CUDA device(s) visible; disabling it.",
                torch.cuda.device_count(),
            )
            use_data_parallel = False
        else:
            model = torch.nn.DataParallel(model)
            logger.info("enabled DataParallel over %d visible CUDA devices", torch.cuda.device_count())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    data = torch.from_numpy(patches)
    if pair_clean_np is not None:
        pair_clean = torch.from_numpy(pair_clean_np)
        pair_attacked = torch.from_numpy(pair_attacked_np)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed)

    n = len(data)
    steps = 0
    for epoch in range(args.epochs):
        perm = torch.randperm(n, generator=rng)
        total = 0.0
        dsm_total = 0.0
        geo_total = 0.0
        chamfer_total = 0.0
        centroid_total = 0.0
        cov_total = 0.0
        density_total = 0.0
        pair_total = 0.0
        count = 0
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            x0 = data[idx].to(device)
            b = x0.shape[0]
            t = torch.rand(b, device=device) * (cfg.t_max - cfg.t_eps) + cfg.t_eps
            abar = torch.exp(-cfg.beta_min * t - 0.5 * (cfg.beta_max - cfg.beta_min) * t * t)
            sigma = torch.sqrt(torch.clamp(1.0 - abar, min=1e-6))
            eps = torch.randn_like(x0)
            xt = torch.sqrt(abar)[:, None, None] * x0 + sigma[:, None, None] * eps
            target = -eps / sigma[:, None, None]

            pred = model(xt, t)
            dsm_loss = score_matching_loss(pred, target, sigma, args.loss_profile)
            pair_loss = x0.new_zeros(())
            if args.loss_profile == "geo":
                x_hat = (xt + (sigma[:, None, None] ** 2) * pred) / torch.sqrt(abar)[:, None, None]
                geo_loss, geo_parts = geometry_consistency_loss(
                    x_hat,
                    x0,
                    lambda_chamfer=args.lambda_chamfer,
                    lambda_centroid=args.lambda_centroid,
                    lambda_cov=args.lambda_cov,
                )
                density_loss = x0.new_zeros(())
                loss = dsm_loss + geo_loss
            elif args.loss_profile == "density":
                x_hat = (xt + (sigma[:, None, None] ** 2) * pred) / torch.sqrt(abar)[:, None, None]
                density_loss = density_spacing_loss(x_hat, x0, args.density_k)
                geo_loss = args.lambda_density * density_loss
                geo_parts = {
                    "chamfer": x0.new_zeros(()),
                    "centroid": x0.new_zeros(()),
                    "cov": x0.new_zeros(()),
                    "geo": geo_loss.detach(),
                }
                loss = dsm_loss + geo_loss
            elif args.loss_profile == "geo_density_pair":
                x_hat = (xt + (sigma[:, None, None] ** 2) * pred) / torch.sqrt(abar)[:, None, None]
                density_loss = density_spacing_loss(x_hat, x0, args.density_k)
                geo_loss = args.lambda_density * density_loss
                geo_parts = {
                    "chamfer": x0.new_zeros(()),
                    "centroid": x0.new_zeros(()),
                    "cov": x0.new_zeros(()),
                    "geo": geo_loss.detach(),
                }
                pair_loss = x0.new_zeros(())
                if pair_clean is not None and args.pair_fraction > 0.0:
                    pair_batch = max(1, int(round(b * args.pair_fraction)))
                    pair_idx = torch.randint(len(pair_clean), (pair_batch,), generator=rng)
                    pair_x0 = pair_clean[pair_idx].to(device)
                    pair_xa = pair_attacked[pair_idx].to(device)
                    pair_t = torch.rand(pair_batch, device=device) * (cfg.t_max - cfg.t_eps) + cfg.t_eps
                    pair_abar = torch.exp(
                        -cfg.beta_min * pair_t
                        - 0.5 * (cfg.beta_max - cfg.beta_min) * pair_t * pair_t
                    )
                    pair_sigma = torch.sqrt(torch.clamp(1.0 - pair_abar, min=1e-6))
                    pair_eps = torch.randn_like(pair_xa)
                    pair_xt = (
                        torch.sqrt(pair_abar)[:, None, None] * pair_xa
                        + pair_sigma[:, None, None] * pair_eps
                    )
                    pair_pred = model(pair_xt, pair_t)
                    pair_xhat = (
                        pair_xt + (pair_sigma[:, None, None] ** 2) * pair_pred
                    ) / torch.sqrt(pair_abar)[:, None, None]
                    pair_loss = torch.nn.functional.smooth_l1_loss(pair_xhat, pair_x0)
                loss = dsm_loss + geo_loss + args.lambda_pair * pair_loss
            else:
                geo_loss = x0.new_zeros(())
                density_loss = geo_loss
                pair_loss = geo_loss
                geo_parts = {
                    "chamfer": geo_loss,
                    "centroid": geo_loss,
                    "cov": geo_loss,
                    "geo": geo_loss,
                }
                loss = dsm_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            total += float(loss.detach().cpu()) * b
            dsm_total += float(dsm_loss.detach().cpu()) * b
            geo_total += float(geo_loss.detach().cpu()) * b
            chamfer_total += float(geo_parts["chamfer"].cpu()) * b
            centroid_total += float(geo_parts["centroid"].cpu()) * b
            cov_total += float(geo_parts["cov"].cpu()) * b
            density_total += float(density_loss.detach().cpu()) * b
            pair_total += float(pair_loss.detach().cpu()) * b
            count += b
            steps += 1
        logger.info(
            "epoch %d/%d loss=%.6f dsm=%.6f geo=%.6f chamfer=%.6f centroid=%.6f cov=%.6f density=%.6f pair=%.6f",
            epoch + 1,
            args.epochs,
            total / max(count, 1),
            dsm_total / max(count, 1),
            geo_total / max(count, 1),
            chamfer_total / max(count, 1),
            centroid_total / max(count, 1),
            cov_total / max(count, 1),
            density_total / max(count, 1),
            pair_total / max(count, 1),
        )

    save_score_net_checkpoint(
        args.out_ckpt,
        model,
        cfg,
        extra={
            "clean_dir": str(Path(args.clean_dir).resolve()),
            "num_patches": int(len(patches)),
            "epochs": int(args.epochs),
            "steps": int(steps),
            "num_features": int(args.num_features),
            "data_parallel": bool(use_data_parallel),
            "loss_profile": str(args.loss_profile),
            "lambda_chamfer": float(args.lambda_chamfer),
            "lambda_centroid": float(args.lambda_centroid),
            "lambda_cov": float(args.lambda_cov),
            "lambda_density": float(args.lambda_density),
            "density_k": int(args.density_k),
            "pair_fraction": float(args.pair_fraction),
            "lambda_pair": float(args.lambda_pair),
            "paired_attacked_dir": str(Path(args.paired_attacked_dir).resolve()) if args.paired_attacked_dir else None,
            "init_ckpt": str(Path(args.init_ckpt).resolve()) if args.init_ckpt else None,
            "note": "Track-A starter local VP-SDE score net",
        },
    )
    logger.info("saved checkpoint: %s", args.out_ckpt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean_dir", default="data/kitti/training/velodyne")
    parser.add_argument("--out_ckpt", required=True)
    parser.add_argument("--max_frames", type=int, default=128)
    parser.add_argument("--patches_per_frame", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--beta_min", type=float, default=0.1)
    parser.add_argument("--beta_max", type=float, default=20.0)
    parser.add_argument("--t_eps", type=float, default=1e-3)
    parser.add_argument("--t_max", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--loss_profile",
        choices=LOSS_PROFILES,
        default="score_mse",
        help=(
            "Training objective. score_mse is the original DSM objective; "
            "time_sigma2 weights per-sample DSM by sigma_t^2; "
            "geo adds geometry consistency on top of time_sigma2; "
            "density adds kNN spacing preservation on top of time_sigma2; "
            "geo_density_pair adds attacked-clean paired Huber fine-tuning."
        ),
    )
    parser.add_argument(
        "--lambda_chamfer",
        type=float,
        default=0.1,
        help="L2 geo profile weight for symmetric Chamfer patch reconstruction.",
    )
    parser.add_argument(
        "--lambda_centroid",
        type=float,
        default=0.05,
        help="L2 geo profile weight for local patch centroid preservation.",
    )
    parser.add_argument(
        "--lambda_cov",
        type=float,
        default=0.05,
        help="L2 geo profile weight for local patch covariance preservation.",
    )
    parser.add_argument(
        "--lambda_density",
        type=float,
        default=0.03,
        help="L3 density profile weight for local kNN spacing preservation.",
    )
    parser.add_argument(
        "--density_k",
        type=int,
        default=8,
        help="Number of nearest neighbors used by the L3 density profile.",
    )
    parser.add_argument(
        "--paired_attacked_dir",
        default=None,
        help="Directory of attacked .bin files paired by filename with --clean_dir for L4.",
    )
    parser.add_argument(
        "--paired_max_frames",
        type=int,
        default=None,
        help="Maximum paired frames for L4; defaults to --max_frames.",
    )
    parser.add_argument(
        "--paired_patches_per_frame",
        type=int,
        default=32,
        help="Number of paired clean/attacked SPU patches sampled per paired frame.",
    )
    parser.add_argument(
        "--pair_fraction",
        type=float,
        default=0.0,
        help="Paired batch fraction used by geo_density_pair fine-tuning.",
    )
    parser.add_argument(
        "--lambda_pair",
        type=float,
        default=0.25,
        help="Weight for attacked-clean paired Huber reconstruction in L4.",
    )
    parser.add_argument(
        "--init_ckpt",
        default=None,
        help="Optional score-net checkpoint used to initialize/fine-tune the model.",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=4,
        help="Point feature dimension in each .bin file: KITTI=4, nuScenes=5.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--data_parallel",
        action="store_true",
        help="Use torch.nn.DataParallel across all visible CUDA devices.",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
