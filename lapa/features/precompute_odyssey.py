"""Precompute DINOv2 features + 2D tracks for PointOdyssey-MC virtual cameras.

Virtual studio views have no RGB. CoTracker therefore runs on the **source**
video; 2D residuals are lifted with source GT depth and reprojected into each
virtual camera. ``tracks_2d_gt`` stays the geometric projection; ``tracks_2d``
is the CoTracker observation used at train/eval time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lapa.data.mc_builder import camera_to_world, in_bounds, project_points
from lapa.features.precompute import (
    DINOv2FeatureExtractor,
    get_cotracker,
    run_cotracker_chunked,
)


def load_rgb_stack(rgb_dir: Path, T: int) -> np.ndarray:
    frames = []
    for t in range(T):
        path = rgb_dir / f"rgb_{t:05d}.jpg"
        if not path.exists():
            alts = sorted(rgb_dir.glob("*.jpg"))
            if t < len(alts):
                path = alts[t]
            else:
                raise FileNotFoundError(path)
        frames.append(np.asarray(Image.open(path).convert("RGB")))
    return np.stack(frames, axis=0)


def backproject(uv: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    z = np.clip(depth, 1e-6, None)
    x = (uv[..., 0] - cx) / max(fx, 1e-6) * z
    y = (uv[..., 1] - cy) / max(fy, 1e-6) * z
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def source_cotracker_tracks(
    frames: np.ndarray,
    tracks_world: np.ndarray,
    src_K: np.ndarray,
    src_w2c: np.ndarray,
    W: int,
    H: int,
    device: torch.device,
    cotracker,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CoTracker on source RGB. Returns src GT uv, CT uv, GT vis, CT vis."""
    T, N, _ = tracks_world.shape
    uv_gt = np.zeros((T, N, 2), dtype=np.float32)
    depth_gt = np.zeros((T, N), dtype=np.float32)
    vis_gt = np.zeros((T, N), dtype=bool)
    for t in range(T):
        uv, depth, valid = project_points(tracks_world[t], src_K[t], src_w2c[t])
        ib = in_bounds(uv, W, H)
        uv_gt[t] = uv.astype(np.float32)
        depth_gt[t] = depth.astype(np.float32)
        vis_gt[t] = valid & ib

    queries = np.zeros((N, 3), dtype=np.float32)
    for m in range(N):
        ts = np.flatnonzero(vis_gt[:, m])
        tq = int(ts[0]) if ts.size else 0
        queries[m, 0] = uv_gt[tq, m, 0]
        queries[m, 1] = uv_gt[tq, m, 1]
        queries[m, 2] = float(tq)

    tracks_2d, vis_ct = run_cotracker_chunked(
        frames, queries, device, model=cotracker, window=window
    )
    bad = ~np.isfinite(tracks_2d).all(axis=-1)
    tracks_2d[bad] = uv_gt[bad]
    for m in range(N):
        tq = int(queries[m, 2])
        if tq > 0:
            tracks_2d[:tq, m] = uv_gt[:tq, m]
            vis_ct[:tq, m] = False
    vis_ct = vis_ct & vis_gt
    return uv_gt, tracks_2d, vis_gt, vis_ct, depth_gt


def reproject_to_virtual(
    src_uv_ct: np.ndarray,
    src_depth_gt: np.ndarray,
    src_vis_ct: np.ndarray,
    src_K: np.ndarray,
    src_w2c: np.ndarray,
    virt_K: np.ndarray,
    virt_w2c: np.ndarray,
    uv_gt_virt: np.ndarray,
    vis_gt_virt: np.ndarray,
    W: int,
    H: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift source CoTracker 2D with GT depth and project into a virtual camera."""
    T, N, _ = src_uv_ct.shape
    uv_obs = uv_gt_virt.copy()
    vis_ct = np.zeros((T, N), dtype=bool)
    for t in range(T):
        pts_cam = backproject(src_uv_ct[t], src_depth_gt[t], src_K[t])
        pts_w = camera_to_world(pts_cam, src_w2c[t])
        uv, depth, valid = project_points(pts_w, virt_K, virt_w2c)
        ib = in_bounds(uv, W, H)
        ok = src_vis_ct[t] & valid & ib & vis_gt_virt[t]
        uv_obs[t, ok] = uv[ok].astype(np.float32)
        vis_ct[t] = ok
    return uv_obs, vis_ct


def _replace_track_datasets(
    out_h5: Path,
    tracks_2d: np.ndarray,
    tracks_2d_gt: np.ndarray,
    vis_gt: np.ndarray,
    vis_ct: np.ndarray,
) -> None:
    with h5py.File(out_h5, "a") as f:
        for name, data in (
            ("tracks_2d", tracks_2d.astype(np.float32)),
            ("tracks_2d_gt", tracks_2d_gt.astype(np.float32)),
            ("visibility", vis_gt.astype(np.uint8)),
            ("visibility_tracker", vis_ct.astype(np.uint8)),
        ):
            if name in f:
                del f[name]
            f.create_dataset(name, data=data, compression="gzip")
        f.attrs["use_cotracker"] = 1


def process_scene(
    scene: str,
    mc_dir: Path,
    out_dir: Path,
    dino: Optional[DINOv2FeatureExtractor],
    device: torch.device,
    cotracker,
    window: int,
    tracks_only: bool,
) -> None:
    meta = json.loads((mc_dir / f"{scene}_mc.json").read_text())
    src = Path(meta["source_seq"])
    rgb_dir = src / "rgbs"
    src_cam = np.load(mc_dir / scene / "source_cameras.npz")
    src_K = np.asarray(src_cam["intrinsics"], dtype=np.float64)
    src_w2c = np.asarray(src_cam["extrinsics"], dtype=np.float64)

    T = int(meta["T"])
    W, H = int(meta["width"]), int(meta["height"])

    print(f"[{scene}] loading {T} RGB frames ...")
    frames = load_rgb_stack(rgb_dir, T)
    assert frames.shape[0] == T

    # Shared world tracks (same identities in every virtual camera)
    cam0 = meta["cameras"][0]
    tw0 = np.load(mc_dir / scene / "tracks_world" / f"cam_{cam0['cam_id']}.npz")
    tracks_world = np.asarray(tw0["tracks_world"], dtype=np.float32)

    print(f"[{scene}] CoTracker on source RGB (window={window}) ...")
    uv_src_gt, src_uv_ct, vis_src_gt, src_vis_ct, src_depth = source_cotracker_tracks(
        frames, tracks_world, src_K, src_w2c, W, H, device, cotracker, window
    )
    src_err = np.linalg.norm(src_uv_ct - uv_src_gt, axis=-1)
    src_mean = float(src_err[vis_src_gt].mean()) if vis_src_gt.any() else float("nan")
    print(f"[{scene}] source |CT-GT| px mean={src_mean:.2f} vis={float(src_vis_ct.mean())*100:.1f}%")
    patch_tokens = None
    if dino is not None and not tracks_only:
        print(f"[{scene}] DINOv2 ...")
        patch_tokens = dino.extract_video_features(frames)

    for cam in meta["cameras"]:
        cam_id = cam["cam_id"]
        out_h5 = out_dir / scene / f"cam_{cam_id}.h5"
        tracks = np.load(mc_dir / scene / "tracks_world" / f"cam_{cam_id}.npz")
        tw = np.asarray(tracks["tracks_world"], dtype=np.float32)
        vis_ann = np.asarray(tracks["visibility"], dtype=bool)
        K = np.asarray(cam["K"], dtype=np.float64)
        w2c = np.asarray(cam["w2c"], dtype=np.float64)

        uv_gt = np.zeros((T, tw.shape[1], 2), dtype=np.float32)
        vis_gt = np.zeros((T, tw.shape[1]), dtype=bool)
        for t in range(T):
            uv, depth, valid = project_points(tw[t], K, w2c)
            ib = in_bounds(uv, W, H)
            uv_gt[t] = uv.astype(np.float32)
            vis_gt[t] = valid & ib & vis_ann[t]

        uv_obs, vis_ct = reproject_to_virtual(
            src_uv_ct, src_depth, src_vis_ct, src_K, src_w2c, K, w2c, uv_gt, vis_gt, W, H
        )
        delta = np.linalg.norm(uv_obs - uv_gt, axis=-1)
        print(
            f"  cam {cam_id} |CT-GT| px mean={float(delta[vis_gt].mean()) if vis_gt.any() else 0:.2f} "
            f"vis_ct={float(vis_ct.mean())*100:.1f}%"
        )
        if vis_gt.any() and float(delta[vis_gt].mean()) < 1e-4:
            raise RuntimeError(
                f"{scene} cam {cam_id}: tracks_2d still matches GT; CoTracker transfer failed"
            )

        if tracks_only and out_h5.exists():
            _replace_track_datasets(out_h5, uv_obs, uv_gt, vis_gt, vis_ct)
            print(f"  updated tracks {out_h5}")
            continue

        if dino is None:
            raise RuntimeError("DINOv2 extractor required unless --tracks_only")
        if patch_tokens is None:
            patch_tokens = dino.extract_video_features(frames)

        src_uv = np.zeros((T, tw.shape[1], 2), dtype=np.float32)
        src_vis = np.zeros((T, tw.shape[1]), dtype=bool)
        for t in range(T):
            uv, depth, valid = project_points(tw[t], src_K[t], src_w2c[t])
            ib = in_bounds(uv, W, H)
            src_uv[t] = uv.astype(np.float32)
            src_vis[t] = valid & ib
        feats = dino.sample_at_points(patch_tokens, src_uv, (W, H)).copy()
        feats[~src_vis] = 0

        out_h5.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_h5.with_suffix(".partial.h5")
        with h5py.File(tmp, "w") as f:
            f.create_dataset("tracks_2d", data=uv_obs, compression="gzip")
            f.create_dataset("tracks_2d_gt", data=uv_gt, compression="gzip")
            f.create_dataset("visibility", data=vis_gt.astype(np.uint8), compression="gzip")
            f.create_dataset("visibility_tracker", data=vis_ct.astype(np.uint8), compression="gzip")
            f.create_dataset("features", data=feats, compression="gzip")
            f.create_dataset("source_uv", data=src_uv, compression="gzip")
            f.attrs["T"] = T
            f.attrs["N"] = tw.shape[1]
            f.attrs["H"] = H
            f.attrs["W"] = W
            f.attrs["dino_dim"] = 768
            f.attrs["scene"] = scene
            f.attrs["cam_id"] = int(cam_id)
            f.attrs["dataset"] = "pointodyssey_mc"
            f.attrs["use_cotracker"] = 1
        tmp.replace(out_h5)
        print(f"  wrote {out_h5}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc_dir", default="./data/pointodyssey_mc")
    parser.add_argument("--out_dir", default="./data/feature_cache_odyssey")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument(
        "--tracks_only",
        action="store_true",
        help="Rewrite tracks_2d in existing HDF5 caches (keep DINOv2 features).",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dino = None if args.tracks_only else DINOv2FeatureExtractor(device)
    print("Loading CoTracker3 ...")
    cotracker = get_cotracker(device)
    mc_dir = Path(args.mc_dir)
    out_dir = Path(args.out_dir)
    index = json.loads((mc_dir / "index.json").read_text())
    scenes = args.scenes or list(index["scenes"].keys())

    for scene in scenes:
        process_scene(
            scene, mc_dir, out_dir, dino, device, cotracker, args.window, args.tracks_only
        )
    print("Done.")


if __name__ == "__main__":
    main()
