#!/usr/bin/env python3
"""Train LAPA on TAPVid-3D-MC with the paper's training recipe.

Loss: L = 1.0 L_recon + 0.7 L_proj + 0.8 L_attn (+ 0.5 L_vis)
Optimizer: AdamW lr=1e-4, wd=1e-5
Schedule: cosine with 5-epoch warmup, 50 epochs, batch size 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lapa.data.mc_dataset import TAPVid3DMCDataset, collate_identity
from lapa.data.odyssey_mc_dataset import PointOdysseyMCDataset
from lapa.data.joint_mc_dataset import JointMCDataset
from lapa.eval.protocol import score_tracks
from lapa.losses import LAPALoss
from lapa.models.lapa import LAPA, count_parameters


def _stack_obs_2d(view_pts_native):
    """List[view][t](M,2) -> list[view] of (T, M, 2)."""
    return [torch.stack(view, dim=0) for view in view_pts_native]


def cosine_warmup_lambda(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return float(epoch + 1) / float(warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(model, loader, criterion, optimizer, device, max_steps=None):
    model.train()
    meters = {"loss": 0.0, "l_recon": 0.0, "l_proj": 0.0, "l_attn": 0.0, "l_vis": 0.0, "n": 0}
    pbar = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(pbar):
        if max_steps is not None and step >= max_steps:
            break
        # Move tensors
        gt_norm = batch["gt_norm"].to(device)
        visible = batch["visible"].to(device)
        queries0 = batch["queries0"].to(device)
        view_K = [k.to(device) for k in batch["view_K"]]
        view_w2c = [w.to(device) for w in batch["view_w2c_norm"]]
        view_pts_native = [
            [p.to(device) for p in view] for view in batch["view_points_2d_native"]
        ]
        view_feats_native = [
            [f.to(device) for f in view] for view in batch["view_features_native"]
        ]
        view_pts_proj = [p.to(device) for p in batch["view_points_2d"]]
        vis_per_view = [v.to(device) for v in batch["visible_per_view"]]
        image_size = tuple(batch["image_size"])
        aabb_center = batch["aabb_center"].to(device)
        aabb_half = batch["aabb_half"].to(device)

        out = model(
            view_pts_native,
            view_feats_native,
            view_K,
            view_w2c,
            queries0,
            image_size,
            view_valid=vis_per_view,
        )
        # Use grid from last frame for attn loss
        grid = model.create_grid(device)
        loss_out = criterion(
            pred_norm=out["points_3d"],
            gt_norm=gt_norm,
            visible=visible,
            gt_points_2d=view_pts_proj,
            view_K=view_K,
            view_w2c_norm=view_w2c,
            visible_per_view=vis_per_view,
            attn_lists=out["attn_lists"],
            grid=grid,
            vis_logits=out["vis_logits"],
            aabb_center=aabb_center,
            aabb_half=aabb_half,
            view_w=out.get("view_w"),
            obs_points_2d=_stack_obs_2d(view_pts_native),
            image_size=image_size,
        )
        if not torch.isfinite(loss_out["loss"]):
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss_out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        meters["n"] += 1
        for k in ("loss", "l_recon", "l_proj", "l_attn", "l_vis"):
            if k in loss_out:
                meters[k] += float(loss_out[k] if k != "loss" else loss_out["loss"])
        pbar.set_postfix(
            loss=f"{meters['loss']/meters['n']:.4f}",
            recon=f"{meters['l_recon']/meters['n']:.4f}",
        )

    n = max(meters["n"], 1)
    return {k: meters[k] / n for k in meters if k != "n"}


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_steps=50):
    model.eval()
    meters = {"loss": 0.0, "l_recon": 0.0, "l_proj": 0.0, "n": 0, "mpjpe": 0.0, "apd": 0.0, "mpjpe_m": 0.0}
    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        gt_norm = batch["gt_norm"].to(device)
        visible = batch["visible"].to(device)
        queries0 = batch["queries0"].to(device)
        view_K = [k.to(device) for k in batch["view_K"]]
        view_w2c = [w.to(device) for w in batch["view_w2c_norm"]]
        view_pts_native = [
            [p.to(device) for p in view] for view in batch["view_points_2d_native"]
        ]
        view_feats_native = [
            [f.to(device) for f in view] for view in batch["view_features_native"]
        ]
        view_pts_proj = [p.to(device) for p in batch["view_points_2d"]]
        vis_per_view = [v.to(device) for v in batch["visible_per_view"]]
        image_size = tuple(batch["image_size"])
        aabb_center = batch["aabb_center"].to(device)
        aabb_half = batch["aabb_half"].to(device)

        out = model(
            view_pts_native,
            view_feats_native,
            view_K,
            view_w2c,
            queries0,
            image_size,
            view_valid=vis_per_view,
        )
        grid = model.create_grid(device)
        loss_out = criterion(
            pred_norm=out["points_3d"],
            gt_norm=gt_norm,
            visible=visible,
            gt_points_2d=view_pts_proj,
            view_K=view_K,
            view_w2c_norm=view_w2c,
            visible_per_view=vis_per_view,
            attn_lists=out["attn_lists"],
            grid=grid,
            vis_logits=out["vis_logits"],
            aabb_center=aabb_center,
            aabb_half=aabb_half,
            view_w=out.get("view_w"),
            obs_points_2d=_stack_obs_2d(view_pts_native),
            image_size=image_size,
        )
        # MPJPE in normalized space
        err = ((out["points_3d"] - gt_norm) ** 2).sum(-1).sqrt()
        w = visible.float()
        mpjpe = (err * w).sum() / w.sum().clamp(min=1)

        meters["n"] += 1
        meters["loss"] += float(loss_out["loss"])
        meters["l_recon"] += float(loss_out["l_recon"])
        meters["l_proj"] += float(loss_out["l_proj"])
        meters["mpjpe"] += float(mpjpe)

        pred_norm = out["points_3d"].detach().cpu().numpy()
        pred_world = pred_norm * aabb_half.detach().cpu().numpy() + aabb_center.detach().cpu().numpy()
        pred_vis = (out["vis_logits"].detach().cpu().numpy() > 0)
        w2c_ref = batch["view_w2c_world"][0].numpy() if "view_w2c_world" in batch else None
        if w2c_ref is None:
            from lapa.eval.protocol import w2c_from_normalized
            w2c_ref = w2c_from_normalized(
                batch["view_w2c_norm"][0].numpy(),
                batch["aabb_center"].numpy(),
                batch["aabb_half"].numpy(),
            )
        K = batch["view_K"][0].cpu().numpy()
        try:
            m = score_tracks(
                pred_world=pred_world,
                gt_world=batch["gt_world"].numpy(),
                pred_visible=pred_vis,
                gt_visible=batch["visible"].numpy(),
                w2c_ref=w2c_ref,
                intrinsics=np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]),
                image_size=image_size,
            )
            meters["apd"] += m["APD"]
            meters["mpjpe_m"] += m.get("MPJPE_m", 0.0)
        except Exception as e:
            print(f"val score_tracks failed: {type(e).__name__}: {e}")

    n = max(meters["n"], 1)
    return {k: meters[k] / n for k in meters if k != "n"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="tapvid3d",
        choices=["tapvid3d", "pointodyssey", "joint"],
        help="Which MC dataset to train on (separate or joint models)",
    )
    parser.add_argument("--mc_dir", default=None)
    parser.add_argument("--feature_dir", default=None)
    parser.add_argument("--tapvid_mc_dir", default="./data/tapvid3d_mc")
    parser.add_argument("--tapvid_feature_dir", default="./data/feature_cache")
    parser.add_argument("--odyssey_mc_dir", default="./data/pointodyssey_mc")
    parser.add_argument("--odyssey_feature_dir", default="./data/feature_cache_odyssey")
    parser.add_argument("--tapvid_prob", type=float, default=0.5)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_frames", type=int, default=24)
    parser.add_argument("--max_points", type=int, default=64)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--volume_size", type=int, default=16)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--val_steps", type=int, default=40)
    parser.add_argument("--overfit", action="store_true", help="Overfit single scene")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_gt_tracks",
        action="store_true",
        help="Use GT 2D at t>0 (overfit / upper bound). Default: CoTracker.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = {
            "pointodyssey": "./checkpoints/lapa_odyssey",
            "joint": "./checkpoints/lapa_joint",
            "tapvid3d": "./checkpoints/lapa",
        }[args.dataset]
    if args.mc_dir is None and args.dataset != "joint":
        args.mc_dir = (
            "./data/pointodyssey_mc"
            if args.dataset == "pointodyssey"
            else "./data/tapvid3d_mc"
        )
    if args.feature_dir is None and args.dataset != "joint":
        args.feature_dir = (
            "./data/feature_cache_odyssey"
            if args.dataset == "pointodyssey"
            else "./data/feature_cache"
        )

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_gt = bool(args.use_gt_tracks or args.overfit)
    overfit_scene = None
    if args.dataset == "joint":
        DatasetCls = JointMCDataset
        ds_kwargs = dict(
            tapvid_mc_dir=args.tapvid_mc_dir,
            tapvid_feature_dir=args.tapvid_feature_dir,
            odyssey_mc_dir=args.odyssey_mc_dir,
            odyssey_feature_dir=args.odyssey_feature_dir,
            data_root=args.data_root,
            num_views=args.num_views,
            num_frames=args.num_frames if not args.overfit else 16,
            max_points=args.max_points if not args.overfit else 32,
            tapvid_prob=args.tapvid_prob,
            use_gt_tracks=use_gt,
        )
    elif args.dataset == "pointodyssey":
        DatasetCls = PointOdysseyMCDataset
        ds_kwargs = dict(
            mc_dir=args.mc_dir,
            feature_dir=args.feature_dir,
            num_views=args.num_views,
            num_frames=args.num_frames if not args.overfit else 16,
            max_points=args.max_points if not args.overfit else 32,
            use_gt_tracks=use_gt,
        )
    else:
        DatasetCls = TAPVid3DMCDataset
        overfit_scene = "boxes"
        ds_kwargs = dict(
            mc_dir=args.mc_dir,
            feature_dir=args.feature_dir,
            data_root=args.data_root,
            num_views=args.num_views,
            num_frames=args.num_frames if not args.overfit else 16,
            max_points=args.max_points if not args.overfit else 32,
            scenes=[overfit_scene] if args.overfit else None,
            use_gt_tracks=use_gt,
        )

    train_ds = DatasetCls(split="train", **ds_kwargs)
    # For overfit, lock to a single sample
    if args.overfit:
        if overfit_scene and overfit_scene in train_ds.scene_cams:
            train_ds.scene_list = [overfit_scene]
        else:
            # Joint / Odyssey: lock first available scene via fixed sample
            if hasattr(train_ds, "scene_list") and train_ds.scene_list:
                pass
        train_ds.fixed_sample = train_ds[0]
        train_ds.length = 50
        args.epochs = min(args.epochs, 100)
        args.steps_per_epoch = 50
        print(f"Overfit locked to scene={train_ds.fixed_sample['scene']} "
              f"cams={train_ds.fixed_sample['cam_ids']}")

    try:
        val_kwargs = dict(ds_kwargs)
        if args.dataset == "tapvid3d":
            val_kwargs["scenes"] = None if not args.overfit else ds_kwargs.get("scenes")
        val_ds = DatasetCls(split="val", **val_kwargs)
    except RuntimeError:
        val_ds = None

    # Match dataset length to steps so tqdm reflects the real epoch size
    train_ds.length = max(args.steps_per_epoch, 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_identity,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_identity,
        )
        if val_ds is not None
        else None
    )

    model = LAPA(volume_size=args.volume_size).to(device)
    nparams = count_parameters(model)
    print(f"LAPA parameters: {nparams:,}")
    criterion = LAPALoss(
        lambda_recon=10.0, lambda_proj=0.5, lambda_attn=0.1, lambda_vis=0.5, lambda_view=2.0
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: cosine_warmup_lambda(e, args.warmup_epochs, args.epochs),
    )

    start_epoch = 0
    best_recon = float("inf")
    best_apd = -1.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_recon = ckpt.get("best_recon", best_recon)
        best_apd = ckpt.get("best_apd", best_apd)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    history = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_steps=args.steps_per_epoch,
        )
        scheduler.step()
        val_stats = {}
        if val_loader is not None and not args.overfit:
            val_stats = evaluate(
                model, val_loader, criterion, device, max_steps=args.val_steps
            )
        elif args.overfit:
            val_stats = evaluate(
                model, train_loader, criterion, device, max_steps=10
            )

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "time_s": time.time() - t0,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch+1}/{args.epochs}  lr={lr:.2e}  "
            f"train_loss={train_stats['loss']:.4f} recon={train_stats['l_recon']:.4f}  "
            f"val_recon={val_stats.get('l_recon', float('nan')):.4f}  "
            f"val_apd={val_stats.get('apd', float('nan')):.2f}  "
            f"val_mpjpe_m={val_stats.get('mpjpe_m', float('nan')):.4f}  "
            f"val_mpjpe={val_stats.get('mpjpe', float('nan')):.4f}"
        )

        recon = val_stats.get("l_recon", train_stats["l_recon"])
        apd = val_stats.get("apd", 0.0)
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_recon": best_recon,
            "best_apd": best_apd,
            "args": vars(args),
            "nparams": nparams,
        }
        torch.save(ckpt, out_dir / "last.pt")
        # Prefer higher APD; fall back to lower recon if APD is unavailable
        improved = apd > best_apd + 1e-6 if apd > 0 else recon < best_recon
        if improved:
            best_apd = max(best_apd, apd)
            best_recon = min(best_recon, recon)
            ckpt["best_recon"] = best_recon
            ckpt["best_apd"] = best_apd
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  saved best.pt (apd={best_apd:.2f} recon={best_recon:.4f})")

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Huber-in-metres gate: 1.5 cm. Old squared-norm 1e-4 ≈ 1.4 cm.
        if args.overfit and train_stats["l_recon"] < 0.015:
            print(f"Overfit gate PASSED (recon={train_stats['l_recon']:.4f} m < 0.015)")
            break

    print(f"Training complete. Best APD={best_apd:.2f} recon={best_recon:.4f}")
    print(f"Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
