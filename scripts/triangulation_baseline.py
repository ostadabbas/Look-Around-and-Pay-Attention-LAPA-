#!/usr/bin/env python3
"""Weighted DLT triangulation baseline on TAPVid-3D-MC minival.

Harness gate: with GT 2D projections (``--use_gt_tracks``) this must score
high APD/AJ. If it does not, the evaluation metric path is still wrong.

Honest baseline: without ``--use_gt_tracks``, uses CoTracker 2D observations
(query frame still GT). A released LAPA checkpoint should beat this.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lapa.data.mc_dataset import TAPVid3DMCEvalDataset, collate_identity
from lapa.eval.protocol import score_tracks, summarize
from lapa.geom.dlt import projection_matrices, triangulate_dlt_irls
from evaluate_lapa import print_summary


def dlt_predict(batch) -> dict:
    uv = torch.stack(
        [torch.stack(view, dim=0) for view in batch["view_points_2d_native"]],
        dim=0,
    )  # (V, T, M, 2)
    valid = torch.stack(batch["visible_per_view"], dim=0).float()  # (V, T, M)
    P = projection_matrices(batch["view_K"], batch["view_w2c_norm"])  # (V, 3, 4)
    T, M = uv.shape[1], uv.shape[2]
    fallback = batch["queries0"].unsqueeze(0).expand(T, -1, -1).reshape(T * M, 3)
    uv_flat = uv.permute(0, 1, 2, 3).reshape(uv.shape[0], T * M, 2)
    w_flat = valid.reshape(valid.shape[0], T * M)
    pts_norm = triangulate_dlt_irls(
        uv_flat, P, w_flat, fallback=fallback, iters=2, sigma_px=5.0
    )
    pts_norm = pts_norm.reshape(T, M, 3).numpy()

    center = batch["aabb_center"].numpy()
    half = batch["aabb_half"].numpy()
    pred_world = pts_norm * half + center
    # DLT has no occlusion head: predict visible iff >=2 views observe the point
    pred_vis = (valid.sum(dim=0) >= 2).numpy()
    return {
        "pred_world": pred_world,
        "pred_visible": pred_vis,
        "gt_world": batch["gt_world"].numpy(),
        "gt_visible": batch["visible"].numpy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc_dir", default="./data/tapvid3d_mc")
    parser.add_argument("--feature_dir", default="./data/feature_cache")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--max_points", type=int, default=256)
    parser.add_argument(
        "--use_gt_tracks",
        action="store_true",
        help="GT 2D projections (harness gate). Default: CoTracker if cached.",
    )
    parser.add_argument("--output", default="./outputs/dlt_baseline.json")
    args = parser.parse_args()

    ds = TAPVid3DMCEvalDataset(
        mc_dir=args.mc_dir,
        feature_dir=args.feature_dir,
        data_root=args.data_root,
        num_views=args.num_views,
        max_points=args.max_points,
        use_gt_tracks=args.use_gt_tracks,
        load_features=False,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_identity)
    print(f"DLT baseline  n={len(ds)}  use_gt_tracks={args.use_gt_tracks}")

    all_metrics = []
    for batch in tqdm(loader, desc="dlt"):
        pred = dlt_predict(batch)
        K = batch["view_K"][0].numpy()
        intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64)
        try:
            sample = score_tracks(
                pred_world=pred["pred_world"],
                gt_world=pred["gt_world"],
                pred_visible=pred["pred_visible"],
                gt_visible=pred["gt_visible"],
                w2c_ref=batch["view_w2c_world"][0].numpy(),
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
    title = "DLT baseline (GT 2D)" if args.use_gt_tracks else "DLT baseline (CoTracker 2D)"
    print_summary(summary, title=title)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dump = {k: v for k, v in summary.items() if k != "per_sample"}
    dump["use_gt_tracks"] = bool(args.use_gt_tracks)
    dump["per_sample"] = summary["per_sample"]
    with open(args.output, "w") as f:
        json.dump(dump, f, indent=2, default=str)
    print(f"Wrote {args.output}")

    if args.use_gt_tracks and summary["APD"] < 50:
        raise SystemExit(
            f"HARNESS GATE FAILED: GT-DLT APD={summary['APD']:.2f} < 50. "
            "Fix the metric path before training."
        )
    if args.use_gt_tracks:
        print("HARNESS GATE PASSED (GT-DLT APD >= 50).")


if __name__ == "__main__":
    main()
