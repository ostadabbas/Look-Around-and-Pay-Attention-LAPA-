"""Precompute CoTracker + DINOv2 for canonical (or eval) corresponded tracks.

For each camera, project the scene canonical world points to t=0, run CoTracker
for the full sequence, and sample DINOv2 at the GT 2D locations.

Also supports ``--mode eval``: for each minival reference camera, track its
official points in that camera and its two companion views.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lapa.data.mc_builder import in_bounds, project_points
from lapa.data.mc_dataset import pick_companion_cameras
from lapa.features.precompute import (
    DINOv2FeatureExtractor,
    decode_images,
    run_cotracker,
)


def _load_frames(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    return decode_images(data["images_jpeg_bytes"])


def _write_h5(
    out_h5: Path,
    tracks_2d: np.ndarray,
    tracks_2d_gt: np.ndarray,
    vis_gt: np.ndarray,
    vis_ct: np.ndarray,
    feats: np.ndarray,
    attrs: dict,
) -> None:
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_h5.with_suffix(".partial.h5")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("tracks_2d", data=tracks_2d.astype(np.float32), compression="gzip")
        f.create_dataset("tracks_2d_gt", data=tracks_2d_gt.astype(np.float32), compression="gzip")
        f.create_dataset("visibility", data=vis_gt.astype(np.uint8), compression="gzip")
        f.create_dataset("visibility_tracker", data=vis_ct.astype(np.uint8), compression="gzip")
        f.create_dataset("features", data=feats.astype(np.float16), compression="gzip")
        for k, v in attrs.items():
            f.attrs[k] = v
    tmp.replace(out_h5)


def process_camera(
    frames: np.ndarray,
    tracks_world: np.ndarray,
    vis_src: np.ndarray,
    cam: dict,
    dino: DINOv2FeatureExtractor,
    device: torch.device,
    use_cotracker: bool,
    patch_tokens: Optional[torch.Tensor] = None,
    cotracker=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T, M, _ = tracks_world.shape
    H, W = frames.shape[1], frames.shape[2]
    K = np.asarray(cam["K"], dtype=np.float64)
    w2c = np.asarray(cam["w2c"], dtype=np.float64)

    uv_gt = np.zeros((T, M, 2), dtype=np.float32)
    vis_gt = np.zeros((T, M), dtype=bool)
    for t in range(T):
        uv, depth, valid = project_points(tracks_world[t], K, w2c)
        ib = in_bounds(uv, W, H)
        uv_gt[t] = uv.astype(np.float32)
        vis_gt[t] = valid & ib & vis_src[t]

    if use_cotracker:
        # Query each point at its first geometrically-visible frame in THIS
        # camera. Always querying at t=0 leaves companions with OOB starts and
        # near-empty tracker visibility, which collapses multi-view DLT.
        queries = np.zeros((M, 3), dtype=np.float32)
        for m in range(M):
            ts = np.flatnonzero(vis_gt[:, m])
            tq = int(ts[0]) if ts.size else 0
            queries[m, 0] = uv_gt[tq, m, 0]
            queries[m, 1] = uv_gt[tq, m, 1]
            queries[m, 2] = float(tq)
        try:
            tracks_2d, vis_ct = run_cotracker(frames, queries, device, model=cotracker)
            bad = ~np.isfinite(tracks_2d).all(axis=-1)
            tracks_2d[bad] = uv_gt[bad]
            # Before the query frame CoTracker has no support; keep GT there
            # only for filling coords, and mark those frames invalid for CT.
            for m in range(M):
                tq = int(queries[m, 2])
                if tq > 0:
                    tracks_2d[:tq, m] = uv_gt[:tq, m]
                    vis_ct[:tq, m] = False
            vis_ct = vis_ct & vis_gt
        except Exception as e:
            print(f"  CoTracker failed cam {cam['cam_id']}: {e}; using GT 2D")
            tracks_2d, vis_ct = uv_gt.copy(), vis_gt.copy()
    else:
        tracks_2d, vis_ct = uv_gt.copy(), vis_gt.copy()

    if patch_tokens is None:
        patch_tokens = dino.extract_video_features(frames)
    feats = dino.sample_at_points(patch_tokens, uv_gt, (W, H))
    feats = feats.copy()
    feats[~vis_gt] = 0
    return tracks_2d, uv_gt, vis_gt, vis_ct, feats


def run_canonical(
    mc_dir: Path,
    out_dir: Path,
    dino: DINOv2FeatureExtractor,
    device: torch.device,
    use_cotracker: bool,
    scenes=None,
    cotracker=None,
) -> None:
    index = json.loads((mc_dir / "index.json").read_text())
    scenes = scenes or list(index["scenes"].keys())
    for scene in scenes:
        canon_path = mc_dir / scene / "canonical.npz"
        if not canon_path.exists():
            from lapa.data.canonical import build_scene_canonical

            build_scene_canonical(scene, mc_dir)
        canon = np.load(canon_path)
        tracks_world = np.asarray(canon["tracks_world"], dtype=np.float32)
        vis_src = np.asarray(canon["visibility"], dtype=bool)
        meta = json.loads((mc_dir / f"{scene}_mc.json").read_text())

        for cam in tqdm(meta["cameras"], desc=f"{scene} cams"):
            out_h5 = out_dir / scene / f"cam_{cam['cam_id']}.h5"
            if out_h5.exists() and out_h5.stat().st_size > 1000:
                continue
            frames = _load_frames(Path(cam["npz_path"]))
            tracks_2d, uv_gt, vis_gt, vis_ct, feats = process_camera(
                frames, tracks_world, vis_src, cam, dino, device, use_cotracker,
                cotracker=cotracker,
            )
            _write_h5(
                out_h5,
                tracks_2d,
                uv_gt,
                vis_gt,
                vis_ct,
                feats,
                {
                    "T": tracks_2d.shape[0],
                    "N": tracks_2d.shape[1],
                    "H": frames.shape[1],
                    "W": frames.shape[2],
                    "dino_dim": 768,
                    "use_cotracker": int(use_cotracker),
                    "scene": scene,
                    "cam_id": int(cam["cam_id"]),
                    "mode": "canonical",
                },
            )


def run_eval(
    mc_dir: Path,
    out_dir: Path,
    data_root: Path,
    dino: DINOv2FeatureExtractor,
    device: torch.device,
    use_cotracker: bool,
    max_points: int,
    cotracker=None,
) -> None:
    minival = set((data_root / "minival_pstudio.txt").read_text().split())
    index = json.loads((mc_dir / "index.json").read_text())
    for fname in tqdm(sorted(minival), desc="eval cache"):
        if not fname.endswith(".npz"):
            continue
        scene, cam_str = fname.replace(".npz", "").split("_")
        cam_id = int(cam_str)
        if scene not in index["scenes"]:
            continue
        meta = json.loads((mc_dir / f"{scene}_mc.json").read_text())
        cams = list(meta["cameras"])
        ref = next((c for c in cams if int(c["cam_id"]) == cam_id), None)
        if ref is None:
            continue
        companions = pick_companion_cameras(ref, cams, k=2)
        chosen = [ref] + companions

        tracks = np.load(mc_dir / scene / "tracks_world" / f"cam_{cam_id}.npz")
        tw = np.asarray(tracks["tracks_world"], dtype=np.float32)
        vis = np.asarray(tracks["visibility"], dtype=bool)
        cand = np.where(vis[0])[0]
        if len(cand) == 0:
            cand = np.arange(tw.shape[1])
        sel = cand[:max_points]
        tw_s = tw[:, sel]
        vis_s = vis[:, sel]

        # Cache per (ref, view) under eval dir
        for cam in chosen:
            out_h5 = out_dir / scene / f"ref{cam_id}_cam{cam['cam_id']}.h5"
            if out_h5.exists() and out_h5.stat().st_size > 1000:
                continue
            frames = _load_frames(Path(cam["npz_path"]))
            tracks_2d, uv_gt, vis_gt, vis_ct, feats = process_camera(
                frames, tw_s, vis_s, cam, dino, device, use_cotracker,
                cotracker=cotracker,
            )
            _write_h5(
                out_h5,
                tracks_2d,
                uv_gt,
                vis_gt,
                vis_ct,
                feats,
                {
                    "T": tracks_2d.shape[0],
                    "N": tracks_2d.shape[1],
                    "H": frames.shape[1],
                    "W": frames.shape[2],
                    "dino_dim": 768,
                    "use_cotracker": int(use_cotracker),
                    "scene": scene,
                    "ref_cam": cam_id,
                    "cam_id": int(cam["cam_id"]),
                    "mode": "eval",
                },
            )
            # point indices live next to the h5 for reproducibility
            np.save(out_h5.with_suffix(".idx.npy"), sel.astype(np.int32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["canonical", "eval"], default="canonical")
    parser.add_argument("--mc_dir", default="./data/tapvid3d_mc")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--use_cotracker", action="store_true")
    parser.add_argument("--max_points", type=int, default=256)
    parser.add_argument("--scenes", nargs="*", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  cotracker={args.use_cotracker}  mode={args.mode}")
    dino = DINOv2FeatureExtractor(device)
    cotracker = None
    if args.use_cotracker:
        from lapa.features.precompute import get_cotracker
        print("Loading CoTracker3 ...")
        cotracker = get_cotracker(device)
    mc_dir = Path(args.mc_dir)
    if args.mode == "canonical":
        out_dir = Path(args.out_dir or "./data/feature_cache_canonical")
        run_canonical(
            mc_dir, out_dir, dino, device, args.use_cotracker, args.scenes,
            cotracker=cotracker,
        )
    else:
        out_dir = Path(args.out_dir or "./data/feature_cache_eval")
        run_eval(
            mc_dir,
            out_dir,
            Path(args.data_root),
            dino,
            device,
            args.use_cotracker,
            args.max_points,
            cotracker=cotracker,
        )
    print("Done.")


if __name__ == "__main__":
    main()
