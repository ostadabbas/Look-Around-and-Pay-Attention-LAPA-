"""Precompute DINOv2 features for PointOdyssey-MC virtual cameras.

For each virtual studio camera we project GT world tracks to 2D. Appearance
features are sampled from the **source** RGB by projecting the same 3D points
into the moving source camera (exact identity features). This avoids storing
warped RGB while preserving geometric multi-view supervision — same training
recipe as TAPVid-3D-MC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lapa.data.mc_builder import in_bounds, project_points
from lapa.features.precompute import DINOv2FeatureExtractor


def load_rgb_stack(rgb_dir: Path, T: int) -> np.ndarray:
    frames = []
    for t in range(T):
        path = rgb_dir / f"rgb_{t:05d}.jpg"
        if not path.exists():
            # some releases use different naming
            alts = sorted(rgb_dir.glob("*.jpg"))
            if t < len(alts):
                path = alts[t]
            else:
                raise FileNotFoundError(path)
        frames.append(np.asarray(Image.open(path).convert("RGB")))
    return np.stack(frames, axis=0)


def process_scene(
    scene: str,
    mc_dir: Path,
    out_dir: Path,
    dino: DINOv2FeatureExtractor,
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

    print(f"[{scene}] DINOv2 ...")
    patch_tokens = dino.extract_video_features(frames)

    for cam in meta["cameras"]:
        cam_id = cam["cam_id"]
        out_h5 = out_dir / scene / f"cam_{cam_id}.h5"
        if out_h5.exists() and out_h5.stat().st_size > 1000:
            print(f"  skip existing {out_h5}")
            continue

        tracks = np.load(mc_dir / scene / "tracks_world" / f"cam_{cam_id}.npz")
        tracks_world = np.asarray(tracks["tracks_world"], dtype=np.float32)
        tracks_2d = np.asarray(tracks["tracks_2d"], dtype=np.float32)
        visibility = np.asarray(tracks["visibility"], dtype=bool)
        N = tracks_world.shape[1]

        # Sample DINO at source-camera projections of the same 3D points
        src_uv = np.zeros((T, N, 2), dtype=np.float32)
        src_vis = np.zeros((T, N), dtype=bool)
        for t in range(T):
            uv, depth, valid = project_points(tracks_world[t], src_K[t], src_w2c[t])
            ib = in_bounds(uv, W, H)
            src_uv[t] = uv.astype(np.float32)
            src_vis[t] = valid & ib

        feats = dino.sample_at_points(patch_tokens, src_uv, (W, H))
        # If a point is not visible in the source frame, zero its feature
        feats = feats.copy()
        feats[~src_vis] = 0

        out_h5.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_h5.with_suffix(".partial.h5")
        with h5py.File(tmp, "w") as f:
            f.create_dataset("tracks_2d", data=tracks_2d, compression="gzip")
            f.create_dataset("tracks_2d_gt", data=tracks_2d, compression="gzip")
            f.create_dataset("visibility", data=visibility.astype(np.uint8), compression="gzip")
            f.create_dataset("features", data=feats, compression="gzip")
            f.create_dataset("source_uv", data=src_uv, compression="gzip")
            f.attrs["T"] = T
            f.attrs["N"] = N
            f.attrs["H"] = H
            f.attrs["W"] = W
            f.attrs["dino_dim"] = 768
            f.attrs["scene"] = scene
            f.attrs["cam_id"] = int(cam_id)
            f.attrs["dataset"] = "pointodyssey_mc"
        tmp.replace(out_h5)
        print(f"  wrote {out_h5}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc_dir", default="./data/pointodyssey_mc")
    parser.add_argument("--out_dir", default="./data/feature_cache_odyssey")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--scenes", nargs="*", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dino = DINOv2FeatureExtractor(device)
    mc_dir = Path(args.mc_dir)
    out_dir = Path(args.out_dir)
    index = json.loads((mc_dir / "index.json").read_text())
    scenes = args.scenes or list(index["scenes"].keys())

    for scene in scenes:
        try:
            process_scene(scene, mc_dir, out_dir, dino)
        except Exception as e:
            print(f"ERROR {scene}: {e}")
            raise
    print("Done.")


if __name__ == "__main__":
    main()
