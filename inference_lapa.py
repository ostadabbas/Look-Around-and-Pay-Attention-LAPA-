#!/usr/bin/env python3
"""Run LAPA inference on a multi-camera TAPVid-3D-MC scene.

Example:
  python inference_lapa.py \
      --checkpoint checkpoints/lapa/best.pt \
      --scene boxes --cameras 5 6 7 \
      --output outputs/inference_boxes.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from lapa.data.mc_builder import project_points
from lapa.models.lapa import LAPA, build_w2c_normalized


@torch.no_grad()
def run_inference(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = LAPA(volume_size=args.volume_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mc_dir = Path(args.mc_dir)
    meta = json.loads((mc_dir / f"{args.scene}_mc.json").read_text())
    aabb = meta["aabb"]
    center = torch.tensor(aabb["center"], dtype=torch.float32, device=device)
    half = torch.tensor(aabb["half"], dtype=torch.float32, device=device)

    cam_map = {c["cam_id"]: c for c in meta["cameras"]}
    cams = [cam_map[int(c)] for c in args.cameras]

    # Load features / tracks
    view_pts = []
    view_feats = []
    view_K = []
    view_w2c = []
    T = None
    for cam in cams:
        feat_path = Path(args.feature_dir) / args.scene / f"cam_{cam['cam_id']}.h5"
        with h5py.File(feat_path, "r") as f:
            tracks = np.asarray(f["tracks_2d_gt" if args.use_gt_tracks else "tracks_2d"], dtype=np.float32)
            feats = np.asarray(f["features"], dtype=np.float32)
            T = tracks.shape[0] if T is None else T
            # Limit points
            N = min(tracks.shape[1], args.max_points)
            pts_t = [torch.from_numpy(tracks[t, :N]).float().to(device) for t in range(T)]
            feats_t = [torch.from_numpy(feats[t, :N]).float().to(device) for t in range(T)]
        view_pts.append(pts_t)
        view_feats.append(feats_t)
        K = torch.tensor(cam["K"], dtype=torch.float32, device=device)
        w2c = torch.tensor(cam["w2c"], dtype=torch.float32, device=device)
        view_K.append(K)
        view_w2c.append(build_w2c_normalized(w2c, center, half))

    # Queries from reference camera GT tracks (first frame visible)
    ref_tracks = np.load(mc_dir / args.scene / "tracks_world" / f"cam_{cams[0]['cam_id']}.npz")
    tracks_norm = ref_tracks["tracks_norm"]
    vis = ref_tracks["visibility"]
    cand = np.where(vis[0])[0]
    if len(cand) == 0:
        cand = np.arange(tracks_norm.shape[1])
    M = min(args.max_points, len(cand))
    sel = cand[:M]
    queries0 = torch.from_numpy(tracks_norm[0, sel]).float().to(device)

    # Optionally truncate frames
    if args.num_frames is not None:
        T = min(T, args.num_frames)
        view_pts = [[p for p in v[:T]] for v in view_pts]
        view_feats = [[f for f in v[:T]] for v in view_feats]

    out = model(view_pts, view_feats, view_K, view_w2c, queries0, (640, 360))
    pred_norm = out["points_3d"].cpu().numpy()
    pred_world = pred_norm * half.cpu().numpy() + center.cpu().numpy()
    vis_prob = torch.sigmoid(out["vis_logits"]).cpu().numpy()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        points_world=pred_world.astype(np.float32),
        points_norm=pred_norm.astype(np.float32),
        visibility=vis_prob.astype(np.float32),
        cam_ids=np.array(args.cameras),
        scene=args.scene,
        point_idx=sel,
    )
    print(f"Wrote {out_path}  shape={pred_world.shape}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", default="boxes")
    parser.add_argument("--cameras", type=int, nargs="+", default=[5, 6, 7])
    parser.add_argument("--mc_dir", default="./data/tapvid3d_mc")
    parser.add_argument("--feature_dir", default="./data/feature_cache")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--volume_size", type=int, default=16)
    parser.add_argument("--max_points", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--use_gt_tracks", action="store_true", default=True)
    parser.add_argument("--output", default="./outputs/inference.npz")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
