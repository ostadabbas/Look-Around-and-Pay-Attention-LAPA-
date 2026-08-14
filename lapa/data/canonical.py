"""Per-scene canonical point sets for TAPVid-3D-MC.

Merges per-camera lifted world tracks into ~512 identities so any camera
triplet shares the same index space. Correspondence is by canonical index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from lapa.data.mc_builder import SCENES, world_to_normalized


def farthest_point_sample(points: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    n = points.shape[0]
    if n <= k:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(n))]
    dist = np.full(n, np.inf, dtype=np.float64)
    for _ in range(k - 1):
        d = np.linalg.norm(points - points[idx[-1]], axis=1)
        dist = np.minimum(dist, d)
        idx.append(int(dist.argmax()))
    return np.array(idx, dtype=np.int64)


def build_scene_canonical(
    scene: str,
    mc_dir: Path,
    max_points: int = 512,
) -> dict:
    meta = json.loads((mc_dir / f"{scene}_mc.json").read_text())
    aabb = {k: np.array(v, dtype=np.float32) for k, v in meta["aabb"].items()}
    cams = meta["cameras"]

    pool_xyz = []
    pool_src = []  # (cam_id, point_idx)
    T_ref = None
    for cam in cams:
        path = mc_dir / scene / "tracks_world" / f"cam_{cam['cam_id']}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        tw = np.asarray(d["tracks_world"], dtype=np.float32)
        vis = np.asarray(d["visibility"], dtype=bool)
        T_ref = tw.shape[0] if T_ref is None else T_ref
        vis0 = vis[0]
        pts = tw[0, vis0]
        src_idx = np.where(vis0)[0]
        for p, si in zip(pts, src_idx):
            pool_xyz.append(p)
            pool_src.append((int(cam["cam_id"]), int(si)))

    if not pool_xyz:
        raise RuntimeError(f"No visible t=0 points for canonical set in {scene}")

    pool_xyz = np.stack(pool_xyz, axis=0)
    sel = farthest_point_sample(pool_xyz, max_points, seed=hash(scene) % (2**31))
    chosen = [pool_src[i] for i in sel]

    T = T_ref
    M = len(chosen)
    tracks_world = np.zeros((T, M, 3), dtype=np.float32)
    visibility = np.zeros((T, M), dtype=bool)
    source_cam = np.zeros(M, dtype=np.int32)
    source_idx = np.zeros(M, dtype=np.int32)

    cache = {}
    for m, (cam_id, pi) in enumerate(chosen):
        if cam_id not in cache:
            d = np.load(mc_dir / scene / "tracks_world" / f"cam_{cam_id}.npz")
            cache[cam_id] = (
                np.asarray(d["tracks_world"], dtype=np.float32),
                np.asarray(d["visibility"], dtype=bool),
            )
        tw, vis = cache[cam_id]
        t_use = min(T, tw.shape[0])
        tracks_world[:t_use, m] = tw[:t_use, pi]
        visibility[:t_use, m] = vis[:t_use, pi]
        source_cam[m] = cam_id
        source_idx[m] = pi

    tracks_norm = world_to_normalized(tracks_world, aabb).astype(np.float32)
    out_path = mc_dir / scene / "canonical.npz"
    np.savez_compressed(
        out_path,
        tracks_world=tracks_world,
        tracks_norm=tracks_norm,
        visibility=visibility,
        source_cam=source_cam,
        source_idx=source_idx,
    )
    report = {
        "scene": scene,
        "path": str(out_path),
        "T": int(T),
        "M": int(M),
        "vis_frac": float(visibility.mean()),
        "n_source_cams": int(len(set(source_cam.tolist()))),
    }
    print(f"[{scene}] canonical M={M} T={T} vis={report['vis_frac']:.3f} -> {out_path}")
    return report


def build_all(
    mc_dir: Path,
    max_points: int = 512,
    scenes: Optional[List[str]] = None,
) -> dict:
    index = json.loads((mc_dir / "index.json").read_text())
    scenes = scenes or list(index["scenes"].keys()) or list(SCENES)
    reports = {}
    for scene in tqdm(scenes, desc="canonical"):
        reports[scene] = build_scene_canonical(scene, mc_dir, max_points=max_points)
    (mc_dir / "canonical_index.json").write_text(json.dumps(reports, indent=2))
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc_dir", default="./data/tapvid3d_mc")
    parser.add_argument("--max_points", type=int, default=512)
    parser.add_argument("--scenes", nargs="*", default=None)
    args = parser.parse_args()
    reports = build_all(Path(args.mc_dir), max_points=args.max_points, scenes=args.scenes)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
