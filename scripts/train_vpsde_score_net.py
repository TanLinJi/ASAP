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
            loss = torch.mean((pred - target) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            total += float(loss.detach().cpu()) * b
            count += b
            steps += 1
        logger.info("epoch %d/%d loss=%.6f", epoch + 1, args.epochs, total / max(count, 1))

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
