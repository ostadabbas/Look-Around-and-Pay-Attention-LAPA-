#!/usr/bin/env python3
"""Evaluate LAPA on TAPVid-3D-MC minival with official TAPVid-3D metrics.

Reports APD (average_pts_within_thresh), OA (occlusion_accuracy),
3D-AJ (average_jaccard), and a 2D-AJ from reprojected tracks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from lapa.data.mc_dataset import TAPVid3DMCDataset, collate_identity
from lapa.eval.metrics import compute_tapvid3d_metrics
from lapa.models.lapa import LAPA
from torch.utils.data import DataLoader


def reproject_to_ref_camera(
    points_norm: np.ndarray,
    aabb_center: np.ndarray,
    aabb_half: np.ndarray,
    w2c_ref: np.ndarray,
) -> np.ndarray:
    """Convert normalized predictions to reference camera frame (x,y,z)."""
    # norm -> world
    pts_w = points_norm * aabb_half + aabb_center
    R = w2c_ref[:3, :3]
    t = w2c_ref[:3, 3]
    return pts_w @ R.T + t


def project_to_2d(points_cam: np.ndarray, fx_fy_cx_cy: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = fx_fy_cx_cy
    z = np.clip(points_cam[..., 2], 1e-6, None)
    u = fx * (points_cam[..., 0] / z) + cx
    v = fy * (points_cam[..., 1] / z) + cy
    return np.stack([u, v], axis=-1)


@torch.no_grad()
def run_eval(args) -> Dict:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = LAPA(volume_size=args.volume_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch')})")

    ds = TAPVid3DMCDataset(
        mc_dir=args.mc_dir,
        feature_dir=args.feature_dir,
        data_root=args.data_root,
        split="val",
        num_views=args.num_views,
        num_frames=args.num_frames,
        max_points=args.max_points,
    )
    ds.length = args.num_samples
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_identity)

    all_metrics = []
    for batch in tqdm(loader, desc="eval"):
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
        image_size = tuple(batch["image_size"])

        out = model(
            view_pts_native,
            view_feats_native,
            view_K,
            view_w2c,
            queries0,
            image_size,
        )
        pred_norm = out["points_3d"].cpu().numpy()  # (T, M, 3)
        gt_n = gt_norm.cpu().numpy()
        vis = visible.cpu().numpy()  # True = visible
        vis_logit = out["vis_logits"].cpu().numpy()
        pred_vis = vis_logit > 0  # (T, M)

        center = batch["aabb_center"].numpy()
        half = batch["aabb_half"].numpy()
        # Reference camera is view 0 — need world-frame w2c
        # Reconstruct from normalized w2c is awkward; use gt_world path
        gt_world = batch["gt_world"].numpy()  # (T, M, 3)
        pred_world = pred_norm * half + center

        # Convert to ref camera frame using view 0's original w2c from scene meta
        # Approximate: use relative transform from world via stored w2c_norm inverse trick
        # Prefer: project via first camera K/w2c from batch — recover world w2c
        # w2c_norm: R_n = R * diag(half), t_n = R@center + t
        # So R = R_n / half (column-wise), t = t_n - R@center
        w2c_n = batch["view_w2c_norm"][0].numpy()
        Rn = w2c_n[:3, :3]
        tn = w2c_n[:3, 3]
        R = Rn / half.reshape(1, 3)
        t = tn - R @ center
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        pred_cam = reproject_to_ref_camera(pred_norm, center, half, w2c)
        gt_cam = reproject_to_ref_camera(gt_n, center, half, w2c)

        # Metrics expect (N, T, 3) with occluded flags
        # Our arrays are (T, M, 3) — transpose to (M, T, 3)
        pred_tr = np.transpose(pred_cam, (1, 0, 2))
        gt_tr = np.transpose(gt_cam, (1, 0, 2))
        gt_occ = np.transpose(~vis, (1, 0))  # True = occluded
        pred_occ = np.transpose(~pred_vis, (1, 0))

        K = batch["view_K"][0].numpy()
        intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64)

        try:
            m = compute_tapvid3d_metrics(
                gt_occluded=gt_occ.astype(bool),
                gt_tracks=gt_tr.astype(np.float64),
                pred_occluded=pred_occ.astype(bool),
                pred_tracks=pred_tr.astype(np.float64),
                intrinsics_params=intrinsics,
                scaling="median",
                order="n t",
            )
            # 2D-AJ: project to 2D and use reproduce_2d-like thresholds via
            # packing z from GT into tracks for 2D metric approximation
            pred_2d = project_to_2d(pred_cam, intrinsics)
            gt_2d = project_to_2d(gt_cam, intrinsics)
            # Build fake 3D with GT depth for 2D evaluation
            pred_2d3 = np.concatenate(
                [pred_2d, gt_cam[..., 2:3]], axis=-1
            )  # use GT depth
            gt_2d3 = np.concatenate([gt_2d, gt_cam[..., 2:3]], axis=-1)
            m2 = compute_tapvid3d_metrics(
                gt_occluded=gt_occ.astype(bool),
                gt_tracks=np.transpose(gt_2d3, (1, 0, 2)).astype(np.float64),
                pred_occluded=pred_occ.astype(bool),
                pred_tracks=np.transpose(pred_2d3, (1, 0, 2)).astype(np.float64),
                intrinsics_params=intrinsics,
                scaling="reproduce_2d",
                order="n t",
            )
            sample = {
                "APD": float(np.mean(m["average_pts_within_thresh"])) * 100,
                "OA": float(np.mean(m["occlusion_accuracy"])) * 100,
                "AJ3D": float(np.mean(m["average_jaccard"])) * 100,
                "AJ2D": float(np.mean(m2["average_jaccard"])) * 100,
                "scene": batch["scene"],
                "cam_ids": batch["cam_ids"],
            }
            all_metrics.append(sample)
        except Exception as e:
            print(f"metric fail: {e}")

    if not all_metrics:
        raise RuntimeError("No metrics computed")

    summary = {
        "APD": float(np.mean([m["APD"] for m in all_metrics])),
        "OA": float(np.mean([m["OA"] for m in all_metrics])),
        "AJ3D": float(np.mean([m["AJ3D"] for m in all_metrics])),
        "AJ2D": float(np.mean([m["AJ2D"] for m in all_metrics])),
        "n_samples": len(all_metrics),
        "per_sample": all_metrics,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mc_dir", default="./data/tapvid3d_mc")
    parser.add_argument("--feature_dir", default="./data/feature_cache")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--volume_size", type=int, default=16)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--num_frames", type=int, default=24)
    parser.add_argument("--max_points", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=60)
    parser.add_argument("--output", default="./outputs/eval_metrics.json")
    args = parser.parse_args()

    summary = run_eval(args)
    print("=== TAPVid-3D-MC Evaluation ===")
    print(f"APD:   {summary['APD']:.2f}")
    print(f"OA:    {summary['OA']:.2f}")
    print(f"3D-AJ: {summary['AJ3D']:.2f}")
    print(f"2D-AJ: {summary['AJ2D']:.2f}")
    print(f"(n={summary['n_samples']})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
