#!/usr/bin/env python3
"""Evaluate LAPA on TAPVid-3D-MC minival with official TAPVid-3D metrics.

Full-sequence protocol: one sample per minival camera (reference) plus two
companion views. Reports APD, OA, 3D-AJ, 2D-AJ (256-scaled pixels), and the
constant-visible OA baseline.

Query-frame 2D is always GT. ``--use_gt_tracks`` uses GT projections at t>0
(upper bound / harness). Default uses CoTracker observations when cached.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lapa.data.mc_dataset import TAPVid3DMCEvalDataset, collate_identity
from lapa.eval.protocol import score_tracks, summarize
from lapa.models.lapa import LAPA


@torch.no_grad()
def predict_batch(model, batch, device) -> Dict[str, np.ndarray]:
    gt_norm = batch["gt_norm"].to(device)
    queries0 = batch["queries0"].to(device)
    view_K = [k.to(device) for k in batch["view_K"]]
    view_w2c = [w.to(device) for w in batch["view_w2c_norm"]]
    view_pts = [
        [p.to(device) for p in view] for view in batch["view_points_2d_native"]
    ]
    view_feats = [
        [f.to(device) for f in view] for view in batch["view_features_native"]
    ]
    view_valid = [v.to(device) for v in batch["visible_per_view"]]
    image_size = tuple(batch["image_size"])

    out = model(
        view_pts,
        view_feats,
        view_K,
        view_w2c,
        queries0,
        image_size,
        view_valid=view_valid,
    )
    pred_norm = out["points_3d"].cpu().numpy()
    vis_logit = out["vis_logits"].cpu().numpy()
    pred_vis = vis_logit > 0
    center = batch["aabb_center"].numpy()
    half = batch["aabb_half"].numpy()
    pred_world = pred_norm * half + center
    return {
        "pred_world": pred_world,
        "pred_visible": pred_vis,
        "gt_world": batch["gt_world"].numpy(),
        "gt_visible": batch["visible"].numpy(),
    }


@torch.no_grad()
def run_eval(args) -> Dict:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = LAPA(volume_size=args.volume_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch')})")

    ds = TAPVid3DMCEvalDataset(
        mc_dir=args.mc_dir,
        feature_dir=args.feature_dir,
        data_root=args.data_root,
        num_views=args.num_views,
        max_points=args.max_points,
        use_gt_tracks=args.use_gt_tracks,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_identity)
    print(f"Minival sequences: {len(ds)}  use_gt_tracks={args.use_gt_tracks}")

    all_metrics = []
    for batch in tqdm(loader, desc="eval"):
        pred = predict_batch(model, batch, device)
        w2c_ref = batch["view_w2c_world"][0].numpy()
        K = batch["view_K"][0].numpy()
        intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64)
        try:
            sample = score_tracks(
                pred_world=pred["pred_world"],
                gt_world=pred["gt_world"],
                pred_visible=pred["pred_visible"],
                gt_visible=pred["gt_visible"],
                w2c_ref=w2c_ref,
                intrinsics=intrinsics,
                image_size=tuple(batch["image_size"]),
            )
            sample["scene"] = batch["scene"]
            sample["cam_ids"] = batch["cam_ids"]
            sample["fname"] = batch.get("fname", "")
            all_metrics.append(sample)
        except Exception as e:
            print(f"metric fail {batch.get('fname')}: {e}")

    summary = summarize(all_metrics)
    return summary


def print_summary(summary: Dict, title: str = "TAPVid-3D-MC Evaluation") -> None:
    print(f"=== {title} ===")
    print(f"APD:          {summary['APD']:.2f}")
    print(f"OA:           {summary['OA']:.2f}")
    print(f"3D-AJ:        {summary['AJ3D']:.2f}")
    print(f"2D-AJ:        {summary['AJ2D']:.2f}")
    print(f"OA const-vis: {summary['OA_const_vis']:.2f}  (always-visible baseline)")
    print(f"(n={summary['n_samples']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mc_dir", default="./data/tapvid3d_mc")
    parser.add_argument("--feature_dir", default="./data/feature_cache")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--volume_size", type=int, default=16)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--max_points", type=int, default=256)
    parser.add_argument(
        "--use_gt_tracks",
        action="store_true",
        help="Use GT 2D projections at t>0 (upper bound). Default: CoTracker.",
    )
    parser.add_argument("--output", default="./outputs/eval_metrics.json")
    args = parser.parse_args()

    summary = run_eval(args)
    print_summary(summary)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dump = {k: v for k, v in summary.items() if k != "per_sample"}
    dump["per_sample"] = [
        {kk: vv for kk, vv in s.items() if kk not in ("pred",)}
        for s in summary["per_sample"]
    ]
    with open(args.output, "w") as f:
        json.dump(dump, f, indent=2, default=str)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
