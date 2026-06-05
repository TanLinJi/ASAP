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
    save_score_net_checkpoint,
    sample_fixed_patch,
)
from asap.geometry import AdaptiveSPUBuilder, SPUConfig

logger = logging.getLogger(__name__)


LOSS_PROFILES = ("score_mse", "time_sigma2", "geo", "density")


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
    elif profile in {"time_sigma2", "geo", "density"}:
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

    device = torch.device(args.device)
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
            else:
                geo_loss = x0.new_zeros(())
                density_loss = geo_loss
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
            count += b
            steps += 1
        logger.info(
            "epoch %d/%d loss=%.6f dsm=%.6f geo=%.6f chamfer=%.6f centroid=%.6f cov=%.6f density=%.6f",
            epoch + 1,
            args.epochs,
            total / max(count, 1),
            dsm_total / max(count, 1),
            geo_total / max(count, 1),
            chamfer_total / max(count, 1),
            centroid_total / max(count, 1),
            cov_total / max(count, 1),
            density_total / max(count, 1),
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
            "pair_fraction": 0.0,
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
            "density adds kNN spacing preservation on top of time_sigma2."
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
