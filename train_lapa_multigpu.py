#!/usr/bin/env python3
"""Multi-GPU LAPA training via per-step data parallelism across V100s.

Each training step draws ``len(devices)`` independent samples, runs forward+backward
on each GPU, averages gradients onto the primary replica, then steps once.
This preserves the paper's batch-size-1 recipe while scaling throughput ~Nx.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lapa.data.mc_dataset import TAPVid3DMCDataset, collate_identity
from lapa.losses import LAPALoss
from lapa.models.lapa import LAPA, count_parameters


def cosine_warmup_lambda(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return float(epoch + 1) / float(warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def move_batch(batch, device):
    return {
        "gt_norm": batch["gt_norm"].to(device),
        "visible": batch["visible"].to(device),
        "queries0": batch["queries0"].to(device),
        "view_K": [k.to(device) for k in batch["view_K"]],
        "view_w2c_norm": [w.to(device) for w in batch["view_w2c_norm"]],
        "view_points_2d_native": [
            [p.to(device) for p in view] for view in batch["view_points_2d_native"]
        ],
        "view_features_native": [
            [f.to(device) for f in view] for view in batch["view_features_native"]
        ],
        "view_points_2d": [p.to(device) for p in batch["view_points_2d"]],
        "visible_per_view": [v.to(device) for v in batch["visible_per_view"]],
        "image_size": tuple(batch["image_size"]),
        "scene": batch.get("scene"),
    }


def forward_loss(model, criterion, batch, device):
    b = move_batch(batch, device)
    out = model(
        b["view_points_2d_native"],
        b["view_features_native"],
        b["view_K"],
        b["view_w2c_norm"],
        b["queries0"],
        b["image_size"],
    )
    grid = model.create_grid(device)
    loss_out = criterion(
        pred_norm=out["points_3d"],
        gt_norm=b["gt_norm"],
        visible=b["visible"],
        gt_points_2d=b["view_points_2d"],
        view_K=b["view_K"],
        view_w2c_norm=b["view_w2c_norm"],
        visible_per_view=b["visible_per_view"],
        attn_lists=out["attn_lists"],
        grid=grid,
        vis_logits=out["vis_logits"],
    )
    err = ((out["points_3d"] - b["gt_norm"]) ** 2).sum(-1).sqrt()
    w = b["visible"].float()
    mpjpe = (err * w).sum() / w.sum().clamp(min=1)
    return loss_out, float(mpjpe)


def sync_replicas(primary: LAPA, replicas: List[LAPA]):
    state = primary.state_dict()
    for r in replicas:
        r.load_state_dict(state)


def average_grads(primary: LAPA, replicas: List[LAPA]):
    """Average gradients from replicas onto primary (in-place)."""
    n = 1 + len(replicas)
    # Scale primary grads
    for p in primary.parameters():
        if p.grad is not None:
            p.grad.mul_(1.0 / n)
    # Add replica grads
    for r in replicas:
        for p_main, p_rep in zip(primary.parameters(), r.parameters()):
            if p_rep.grad is None:
                continue
            if p_main.grad is None:
                p_main.grad = p_rep.grad.detach().to(p_main.device) / n
            else:
                p_main.grad.add_(p_rep.grad.detach().to(p_main.device) / n)
        # Clear replica grads
        r.zero_grad(set_to_none=True)


def _worker_forward_backward(model, criterion, batch, device, scale, out_box, err_box):
    """Run on a worker thread bound to ``device``."""
    try:
        torch.cuda.set_device(device)
        loss_out, _ = forward_loss(model, criterion, batch, device)
        (loss_out["loss"] * scale).backward()
        # Detach scalars for logging on host
        out_box[0] = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in loss_out.items()}
    except Exception as e:
        err_box[0] = e


def train_one_epoch(
    primary,
    replicas,
    devices,
    loader,
    criterion,
    optimizer,
    max_steps,
):
    import threading

    primary.train()
    for r in replicas:
        r.train()
    meters = {"loss": 0.0, "l_recon": 0.0, "l_proj": 0.0, "l_attn": 0.0, "l_vis": 0.0, "n": 0}
    it = iter(loader)
    models = [primary] + list(replicas)
    scale = 1.0 / len(devices)
    pbar = tqdm(range(max_steps), desc="train", leave=False)
    for _ in pbar:
        batches = []
        for _d in devices:
            try:
                batches.append(next(it))
            except StopIteration:
                it = iter(loader)
                batches.append(next(it))

        optimizer.zero_grad(set_to_none=True)
        for r in replicas:
            r.zero_grad(set_to_none=True)

        # Parallel forward+backward across GPUs
        out_boxes = [[None] for _ in devices]
        err_boxes = [[None] for _ in devices]
        threads = []
        for i, (model, batch, dev) in enumerate(zip(models, batches, devices)):
            t = threading.Thread(
                target=_worker_forward_backward,
                args=(model, criterion, batch, dev, scale, out_boxes[i], err_boxes[i]),
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        for eb in err_boxes:
            if eb[0] is not None:
                raise eb[0]

        average_grads(primary, replicas)
        torch.nn.utils.clip_grad_norm_(primary.parameters(), 1.0)
        optimizer.step()
        sync_replicas(primary, replicas)

        step_stats = [ob[0] for ob in out_boxes]
        meters["n"] += 1
        for lo in step_stats:
            meters["loss"] += float(lo["loss"]) / len(step_stats)
            for k in ("l_recon", "l_proj", "l_attn", "l_vis"):
                if k in lo:
                    meters[k] += float(lo[k]) / len(step_stats)
        pbar.set_postfix(loss=f"{meters['loss']/meters['n']:.4f}")

    n = max(meters["n"], 1)
    return {k: meters[k] / n for k in meters if k != "n"}


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_steps=40):
    model.eval()
    meters = {"loss": 0.0, "l_recon": 0.0, "l_proj": 0.0, "mpjpe": 0.0, "n": 0}
    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        loss_out, mpjpe = forward_loss(model, criterion, batch, device)
        meters["n"] += 1
        meters["loss"] += float(loss_out["loss"])
        meters["l_recon"] += float(loss_out["l_recon"])
        meters["l_proj"] += float(loss_out["l_proj"])
        meters["mpjpe"] += mpjpe
    n = max(meters["n"], 1)
    return {k: meters[k] / n for k in meters if k != "n"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc_dir", default="./data/tapvid3d_mc")
    parser.add_argument("--feature_dir", default="./data/feature_cache")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--output_dir", default="./checkpoints/lapa_mgpu")
    parser.add_argument("--devices", type=str, default="0,1,2,3",
                        help="Comma-separated CUDA device indices (visible devices)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_frames", type=int, default=24)
    parser.add_argument("--max_points", type=int, default=64)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--volume_size", type=int, default=16)
    parser.add_argument("--steps_per_epoch", type=int, default=400)
    parser.add_argument("--val_steps", type=int, default=60)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device_ids = [int(x) for x in args.devices.split(",") if x.strip() != ""]
    devices = [torch.device(f"cuda:{i}") for i in device_ids]
    primary_dev = devices[0]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TAPVid3DMCDataset(
        mc_dir=args.mc_dir,
        feature_dir=args.feature_dir,
        data_root=args.data_root,
        split="train",
        num_views=args.num_views,
        num_frames=args.num_frames,
        max_points=args.max_points,
    )
    train_ds.length = max(args.steps_per_epoch * len(devices) * 2, 2000)
    try:
        val_ds = TAPVid3DMCDataset(
            mc_dir=args.mc_dir,
            feature_dir=args.feature_dir,
            data_root=args.data_root,
            split="val",
            num_views=args.num_views,
            num_frames=args.num_frames,
            max_points=args.max_points,
        )
    except RuntimeError:
        val_ds = None

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_identity
    )
    val_loader = (
        DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_identity)
        if val_ds is not None
        else None
    )

    primary = LAPA(volume_size=args.volume_size).to(primary_dev)
    replicas = []
    for dev in devices[1:]:
        rep = LAPA(volume_size=args.volume_size).to(dev)
        rep.load_state_dict(primary.state_dict())
        replicas.append(rep)

    nparams = count_parameters(primary)
    print(f"LAPA parameters: {nparams:,}")
    print(f"Multi-GPU devices: {devices}  (effective batch={len(devices)})")

    criterion = LAPALoss(lambda_recon=1.0, lambda_proj=0.7, lambda_attn=0.8, lambda_vis=0.5)
    # Scale LR mildly with effective batch
    lr = args.lr * math.sqrt(len(devices))
    optimizer = torch.optim.AdamW(primary.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: cosine_warmup_lambda(e, args.warmup_epochs, args.epochs),
    )

    start_epoch = 0
    best_recon = float("inf")
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=primary_dev)
        primary.load_state_dict(ckpt["model"])
        sync_replicas(primary, replicas)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_recon = ckpt.get("best_recon", best_recon)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    history = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_stats = train_one_epoch(
            primary, replicas, devices, train_loader, criterion, optimizer, args.steps_per_epoch
        )
        scheduler.step()
        val_stats = {}
        if val_loader is not None:
            val_stats = evaluate(primary, val_loader, criterion, primary_dev, args.val_steps)

        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch + 1,
            "lr": lr_now,
            "time_s": time.time() - t0,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch+1}/{args.epochs}  lr={lr_now:.2e}  "
            f"train_loss={train_stats['loss']:.4f} recon={train_stats['l_recon']:.4f}  "
            f"val_recon={val_stats.get('l_recon', float('nan')):.4f}  "
            f"val_mpjpe={val_stats.get('mpjpe', float('nan')):.4f}  "
            f"({row['time_s']:.1f}s)"
        )

        ckpt = {
            "epoch": epoch + 1,
            "model": primary.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_recon": best_recon,
            "args": vars(args),
            "nparams": nparams,
        }
        torch.save(ckpt, out_dir / "last.pt")
        recon = val_stats.get("l_recon", train_stats["l_recon"])
        if recon < best_recon:
            best_recon = recon
            ckpt["best_recon"] = best_recon
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  saved best.pt (recon={best_recon:.4f})")
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"Training complete. Best recon={best_recon:.4f}")


if __name__ == "__main__":
    main()
