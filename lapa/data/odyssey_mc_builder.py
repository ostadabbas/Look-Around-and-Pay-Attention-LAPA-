"""Build PointOdyssey-MC from the robots subset.

Paper recipe (suppl. S4.4): re-engineer PointOdyssey generation for per-camera
calibration + consistent point IDs. Without the authors' Blender fork we
approximate the same geometry by placing a fixed 3-camera studio rig around
each robot sequence (world-space tracks already provided) and projecting
annotations into every view — matching the TAPVid-3D-MC correspondence protocol.

RGB for virtual views is depth-warped from the source camera at feature time
(see ``lapa.features.precompute_odyssey``); this module only writes calibration,
world tracks, and MC metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from lapa.data.mc_builder import (
    compute_scene_aabb,
    in_bounds,
    project_points,
    world_to_normalized,
)


def cam_center_from_w2c(w2c: np.ndarray) -> np.ndarray:
    R, t = w2c[:3, :3], w2c[:3, 3]
    return (-R.T @ t).astype(np.float64)


def look_at_w2c(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float64),
) -> np.ndarray:
    """Build world-to-camera (OpenCV: +X right, +Y down, +Z forward)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    # OpenCV camera: +X right, +Y down, +Z forward. PointOdyssey world is Y-up.
    z = target - eye
    z = z / (np.linalg.norm(z) + 1e-8)
    x = np.cross(up, z)  # right
    if np.linalg.norm(x) < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)  # down

    R = np.stack([x, y, z], axis=0)  # rows = camera axes in world
    t = -R @ eye
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c


def make_studio_cameras(
    aabb: dict,
    K: np.ndarray,
    num_views: int = 3,
    radius_scale: float = 2.2,
    height_frac: float = 0.35,
) -> List[np.ndarray]:
    """Place ``num_views`` cameras on a ring around the scene AABB center."""
    center = aabb["center"].astype(np.float64)
    half = aabb["half"].astype(np.float64)
    radius = float(np.linalg.norm(half[[0, 2]]) * radius_scale)
    radius = max(radius, 1.5)
    height = float(center[1] + height_frac * half[1])

    w2cs = []
    for i in range(num_views):
        ang = 2.0 * np.pi * i / num_views + np.pi / 6.0
        eye = np.array(
            [
                center[0] + radius * np.cos(ang),
                height,
                center[2] + radius * np.sin(ang),
            ],
            dtype=np.float64,
        )
        w2cs.append(look_at_w2c(eye, center))
    return w2cs


def load_odyssey_sequence(seq_dir: Path) -> dict:
    anno = np.load(seq_dir / "anno.npz")
    trajs_3d = np.asarray(anno["trajs_3d"], dtype=np.float32)  # (T, N, 3) world
    trajs_2d = np.asarray(anno["trajs_2d"], dtype=np.float32)
    visibs = np.asarray(anno["visibs"], dtype=bool)
    valids = np.asarray(anno["valids"], dtype=bool) if "valids" in anno.files else np.ones_like(visibs)
    intrinsics = np.asarray(anno["intrinsics"], dtype=np.float64)  # (T, 3, 3)
    extrinsics = np.asarray(anno["extrinsics"], dtype=np.float64)  # (T, 4, 4) w2c

    visibility = visibs & valids
    K0 = intrinsics[0]
    width = int(round(K0[0, 2] * 2))
    height = int(round(K0[1, 2] * 2))
    # Prefer actual RGB size if present
    rgb_dir = seq_dir / "rgbs"
    if rgb_dir.exists():
        rgbs = sorted(rgb_dir.glob("*.jpg"))
        if rgbs:
            from PIL import Image

            with Image.open(rgbs[0]) as im:
                width, height = im.size

    return {
        "seq_dir": seq_dir,
        "tracks_world": trajs_3d,
        "tracks_2d_src": trajs_2d,
        "visibility_src": visibility,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "K": K0,
        "width": width,
        "height": height,
        "T": trajs_3d.shape[0],
        "N": trajs_3d.shape[1],
    }


def subsample_points(
    tracks_world: np.ndarray,
    visibility: np.ndarray,
    max_points: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """Keep points visible often enough; cap at max_points."""
    T, N, _ = tracks_world.shape
    frac = visibility.mean(axis=0)
    cand = np.where(frac > 0.05)[0]
    if len(cand) == 0:
        cand = np.arange(N)
    rng = np.random.default_rng(seed)
    if len(cand) > max_points:
        # Prefer high-visibility points
        scores = frac[cand]
        # weighted sample without replacement via top-k noise
        noise = rng.random(len(cand))
        order = np.argsort(-(scores + 0.05 * noise))
        cand = cand[order[:max_points]]
    cand.sort()
    return cand.astype(np.int64)


def verify_odyssey_calib(seq: dict, w2cs: List[np.ndarray], K: np.ndarray) -> dict:
    """Self-reproj of source cam + virtual-cam in-bounds rates."""
    # Source self-reproj at t=0
    t = 0
    vis = seq["visibility_src"][t]
    pts = seq["tracks_world"][t]
    uv, depth, valid = project_points(pts, seq["intrinsics"][t], seq["extrinsics"][t])
    mask = vis & valid
    err = np.linalg.norm(uv[mask] - seq["tracks_2d_src"][t, mask], axis=-1)
    self_px = float(np.median(err)) if err.size else 1e9

    cross = []
    frames = np.linspace(0, seq["T"] - 1, num=min(8, seq["T"]), dtype=int)
    for w2c in w2cs:
        rates = []
        for f in frames:
            vis_f = seq["visibility_src"][f]
            if vis_f.sum() == 0:
                continue
            uv_v, d_v, valid_v = project_points(
                seq["tracks_world"][f, vis_f], K, w2c
            )
            ib = in_bounds(uv_v, seq["width"], seq["height"]) & valid_v
            rates.append(float(ib.mean()))
        if rates:
            cross.append(float(np.mean(rates)))

    return {
        "median_self_reproj_px": self_px,
        "mean_virtual_in_bounds_frac": float(np.mean(cross)) if cross else 0.0,
        "pass_self_reproj": self_px < 1.0,
        "pass_virtual_coverage": (float(np.mean(cross)) if cross else 0.0) > 0.15,
    }


def build_sequence(
    seq_dir: Path,
    out_dir: Path,
    num_views: int = 3,
    max_points: int = 256,
) -> Optional[dict]:
    scene = seq_dir.name
    seq = load_odyssey_sequence(seq_dir)

    # AABB from source-visible world points
    fake_clip = {
        "tracks_world": seq["tracks_world"],
        "visibility": seq["visibility_src"],
    }
    aabb = compute_scene_aabb([fake_clip])
    K = seq["K"].astype(np.float64)
    w2cs = make_studio_cameras(aabb, K, num_views=num_views)

    calib = verify_odyssey_calib(seq, w2cs, K)
    print(f"[{scene}] calib: {calib}")

    # Subsample points for manageable feature cache
    point_idx = subsample_points(
        seq["tracks_world"], seq["visibility_src"], max_points=max_points, seed=hash(scene) % (2**31)
    )
    tracks_world = seq["tracks_world"][:, point_idx]
    vis_src = seq["visibility_src"][:, point_idx]
    tracks_norm = world_to_normalized(tracks_world, aabb).astype(np.float32)

    scene_out = out_dir / scene
    tracks_dir = scene_out / "tracks_world"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    cameras = []
    for cam_id, w2c in enumerate(w2cs):
        # Per-view visibility via frustum (depth occlusion deferred to precompute)
        T, N, _ = tracks_world.shape
        vis_cam = np.zeros((T, N), dtype=bool)
        tracks_2d = np.zeros((T, N, 2), dtype=np.float32)
        for t in range(T):
            uv, depth, valid = project_points(tracks_world[t], K, w2c)
            ib = in_bounds(uv, seq["width"], seq["height"])
            vis_cam[t] = valid & ib & vis_src[t]
            tracks_2d[t] = uv.astype(np.float32)

        np.savez_compressed(
            tracks_dir / f"cam_{cam_id}.npz",
            tracks_world=tracks_world,
            tracks_norm=tracks_norm,
            tracks_2d=tracks_2d,
            visibility=vis_cam,
            point_idx=point_idx,
        )
        cameras.append(
            {
                "cam_id": cam_id,
                "K": K.tolist(),
                "w2c": w2c.tolist(),
                "width": seq["width"],
                "height": seq["height"],
                "source_seq": str(seq_dir),
                "tracks_path": str(tracks_dir / f"cam_{cam_id}.npz"),
                # Keep source extrinsics path for warping
                "source_extrinsics_key": "extrinsics",
                "source_intrinsics_key": "intrinsics",
            }
        )

    # Also save source camera trajectory for warping
    np.savez_compressed(
        scene_out / "source_cameras.npz",
        intrinsics=seq["intrinsics"],
        extrinsics=seq["extrinsics"],
        K0=K,
    )

    meta = {
        "scene": scene,
        "dataset": "pointodyssey_mc",
        "num_views": num_views,
        "T": int(seq["T"]),
        "N": int(point_idx.shape[0]),
        "N_full": int(seq["N"]),
        "width": seq["width"],
        "height": seq["height"],
        "aabb": {k: v.tolist() for k, v in aabb.items()},
        "cameras": cameras,
        "source_seq": str(seq_dir),
        "calib": calib,
        "split_hint": "val" if scene.startswith("r") and "_f" not in scene.replace("r1_new_f", "") else "train",
    }
    # Simple split: scenes ending with _f or longer variants → train-ish; keep explicit later
    (out_dir / f"{scene}_mc.json").write_text(json.dumps(meta, indent=2))
    return meta


def discover_robot_sequences(raw_root: Path) -> List[Path]:
    seqs = []
    for p in sorted(raw_root.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        # robots: r1_new_f, r4_new_f, etc.
        if name.startswith("r") and (p / "anno.npz").exists():
            seqs.append(p)
    return seqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", default="./data/pointodyssey/raw")
    parser.add_argument("--out_dir", default="./data/pointodyssey_mc")
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--max_points", type=int, default=256)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = discover_robot_sequences(raw_root)
    print(f"Found {len(seqs)} robot sequences under {raw_root}")
    index = {"scenes": {}, "dataset": "pointodyssey_mc"}
    for seq_dir in tqdm(seqs):
        meta = build_sequence(
            seq_dir, out_dir, num_views=args.num_views, max_points=args.max_points
        )
        if meta is None:
            continue
        index["scenes"][meta["scene"]] = {
            "meta": f"{meta['scene']}_mc.json",
            "T": meta["T"],
            "N": meta["N"],
            "calib_pass": bool(
                meta["calib"]["pass_self_reproj"] and meta["calib"]["pass_virtual_coverage"]
            ),
        }

    # Explicit train/val split files (camera-free; sequence-level)
    scenes = sorted(index["scenes"].keys())
    # Hold out the first sequence alphabetically as val if >=2; else reuse
    if len(scenes) >= 2:
        val = [scenes[0]]
        train = scenes[1:]
    elif scenes:
        val = scenes
        train = scenes
    else:
        val, train = [], []
    (out_dir / "train_scenes.txt").write_text("\n".join(train) + ("\n" if train else ""))
    (out_dir / "val_scenes.txt").write_text("\n".join(val) + ("\n" if val else ""))
    index["train_scenes"] = train
    index["val_scenes"] = val
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote index with {len(scenes)} scenes → {out_dir}")
    print(f"train={train} val={val}")


if __name__ == "__main__":
    main()
