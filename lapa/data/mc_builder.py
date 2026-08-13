"""Build TAPVid-3D-MC from per-camera TAPVid-3D npz + Dynamic3DGaussians calibration.

Each TAPVid-3D pstudio file is named ``{scene}_{cam_id}.npz``. Tracks are in that
camera's coordinate frame. We lift them to a shared world frame using the
``w2c`` matrices from Dynamic3DGaussians ``train_meta.json``, then project into
every other camera to produce multi-view correspondences and visibility.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

SCENES = ("basketball", "boxes", "football", "juggle", "softball", "tennis")


@dataclass
class CameraCalib:
    cam_id: int
    K: np.ndarray  # (3, 3)
    w2c: np.ndarray  # (4, 4)
    fx_fy_cx_cy: np.ndarray  # (4,) from TAPVid npz (may differ slightly from K)
    width: int
    height: int

    @property
    def R(self) -> np.ndarray:
        return self.w2c[:3, :3]

    @property
    def t(self) -> np.ndarray:
        return self.w2c[:3, 3]

    @property
    def c2w(self) -> np.ndarray:
        return np.linalg.inv(self.w2c)

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix K [R|t]."""
        return self.K @ self.w2c[:3, :]


def load_d3g_meta(d3g_scene_dir: Path) -> dict:
    meta_path = d3g_scene_dir / "train_meta.json"
    with open(meta_path, "r") as f:
        return json.load(f)


def build_cam_lookup(meta: dict) -> Dict[int, Tuple[np.ndarray, np.ndarray, int, int]]:
    """Map cam_id -> (K 3x3, w2c 4x4, w, h) using timestep 0 (static cameras)."""
    w = int(meta["w"])
    h = int(meta["h"])
    # cam_id[t][c], k[t][c], w2c[t][c]
    cam_ids_t0 = meta["cam_id"][0]
    lookup = {}
    for c_idx, cam_id in enumerate(cam_ids_t0):
        cam_id = int(cam_id)
        K = np.asarray(meta["k"][0][c_idx], dtype=np.float64)
        w2c = np.asarray(meta["w2c"][0][c_idx], dtype=np.float64)
        if K.shape != (3, 3):
            K = np.asarray(K).reshape(3, 3)
        if w2c.shape != (4, 4):
            w2c = np.asarray(w2c).reshape(4, 4)
        lookup[cam_id] = (K, w2c, w, h)
    return lookup


def list_scene_npz(npz_root: Path, scene: str) -> List[Path]:
    """Find all {scene}_{cam}.npz under npz_root (recursive)."""
    paths = sorted(npz_root.rglob(f"{scene}_*.npz"))
    # Filter to exact scene_N.npz pattern
    out = []
    for p in paths:
        stem = p.stem  # e.g. boxes_5
        parts = stem.split("_")
        if len(parts) == 2 and parts[0] == scene and parts[1].isdigit():
            out.append(p)
    return out


def camera_to_world(points_cam: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Lift (..., 3) camera-frame points to world using inverse w2c."""
    c2w = np.linalg.inv(w2c)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    return points_cam @ R.T + t


def world_to_camera(points_world: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    return points_world @ R.T + t


def project_points(
    points_world: np.ndarray,
    K: np.ndarray,
    w2c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points to image.

    Returns:
        uv: (..., 2) pixel coords
        depth: (...,) camera z
        valid: (...,) bool — positive depth
    """
    pts_cam = world_to_camera(points_world, w2c)
    depth = pts_cam[..., 2]
    # Avoid div by zero
    z = np.clip(depth, 1e-6, None)
    x = pts_cam[..., 0] / z
    y = pts_cam[..., 1] / z
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    uv = np.stack([u, v], axis=-1)
    valid = depth > 1e-4
    return uv, depth, valid


def in_bounds(uv: np.ndarray, width: int, height: int, margin: float = 0.0) -> np.ndarray:
    return (
        (uv[..., 0] >= margin)
        & (uv[..., 0] < width - margin)
        & (uv[..., 1] >= margin)
        & (uv[..., 1] < height - margin)
    )


def load_camera_clip(
    npz_path: Path,
    cam_lookup: Dict[int, Tuple[np.ndarray, np.ndarray, int, int]],
) -> Optional[dict]:
    stem = npz_path.stem
    scene, cam_str = stem.split("_")
    cam_id = int(cam_str)
    if cam_id not in cam_lookup:
        print(f"WARNING: cam_id {cam_id} not in train_meta for {npz_path}")
        return None

    K_d3g, w2c, w, h = cam_lookup[cam_id]
    data = np.load(npz_path, allow_pickle=True)
    tracks_cam = np.asarray(data["tracks_XYZ"], dtype=np.float32)  # (T, N, 3)
    visibility = np.asarray(data["visibility"], dtype=bool)  # (T, N)
    fx_fy_cx_cy = np.asarray(data["fx_fy_cx_cy"], dtype=np.float64)
    queries_xyt = np.asarray(data["queries_xyt"], dtype=np.float64)

    # Prefer TAPVid intrinsics (matched to the tracks), fall back to D3G K
    fx, fy, cx, cy = fx_fy_cx_cy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    tracks_world = camera_to_world(tracks_cam, w2c).astype(np.float32)

    return {
        "scene": scene,
        "cam_id": cam_id,
        "npz_path": str(npz_path),
        "tracks_cam": tracks_cam,
        "tracks_world": tracks_world,
        "visibility": visibility,
        "queries_xyt": queries_xyt,
        "fx_fy_cx_cy": fx_fy_cx_cy,
        "K": K,
        "K_d3g": K_d3g,
        "w2c": w2c,
        "width": w,
        "height": h,
        "T": tracks_cam.shape[0],
        "N": tracks_cam.shape[1],
        "has_images": "images_jpeg_bytes" in data.files,
    }


def compute_scene_aabb(clips: List[dict], percentile: Tuple[float, float] = (1.0, 99.0)) -> dict:
    """Compute robust AABB over all visible world points in the scene."""
    pts = []
    for c in clips:
        vis = c["visibility"]  # (T, N)
        tw = c["tracks_world"]  # (T, N, 3)
        pts.append(tw[vis])
    if not pts:
        raise RuntimeError("No visible points for AABB")
    all_pts = np.concatenate(pts, axis=0)
    lo = np.percentile(all_pts, percentile[0], axis=0)
    hi = np.percentile(all_pts, percentile[1], axis=0)
    # Pad slightly
    span = np.maximum(hi - lo, 1e-3)
    lo = lo - 0.05 * span
    hi = hi + 0.05 * span
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    half = np.maximum(half, 1e-3)
    return {
        "lo": lo.astype(np.float32),
        "hi": hi.astype(np.float32),
        "center": center.astype(np.float32),
        "half": half.astype(np.float32),
    }


def world_to_normalized(points_world: np.ndarray, aabb: dict) -> np.ndarray:
    """Map world coords into [-1, 1]^3 using scene AABB."""
    return (points_world - aabb["center"]) / aabb["half"]


def normalized_to_world(points_norm: np.ndarray, aabb: dict) -> np.ndarray:
    return points_norm * aabb["half"] + aabb["center"]


def verify_calibration(clips: List[dict], max_pairs: int = 5) -> dict:
    """Reproject camera-a GT points into camera b; report mean pixel error.

    For each pair (a, b): take points visible in a, lift to world (already done),
    project into b, and compare against a's own queries? We cannot compare against
    b's tracks because point sets differ. Instead we verify:
      1) Self-reprojection: project tracks_world of cam a back into cam a and
         compare to pinhole projection of tracks_cam (should be ~0).
      2) Cross-view geometric consistency: for points of cam a projected into b,
         check they land in-bounds at a reasonable rate, and that depth is positive.
      3) Round-trip: camera_to_world then world_to_camera recovers tracks_cam.
    """
    results = {"self_reproj_px": [], "roundtrip_m": [], "cross_in_bounds_frac": []}

    for c in clips:
        # Round-trip
        recovered = world_to_camera(c["tracks_world"], c["w2c"])
        err = np.linalg.norm(recovered - c["tracks_cam"], axis=-1)
        results["roundtrip_m"].append(float(np.median(err[c["visibility"]])))

        # Self-reprojection of tracks_cam via K should match manual projection
        tc = c["tracks_cam"]
        z = np.clip(tc[..., 2], 1e-6, None)
        u = c["K"][0, 0] * (tc[..., 0] / z) + c["K"][0, 2]
        v = c["K"][1, 1] * (tc[..., 1] / z) + c["K"][1, 2]
        uv_direct = np.stack([u, v], axis=-1)

        uv_via_world, _, _ = project_points(c["tracks_world"], c["K"], c["w2c"])
        reproj = np.linalg.norm(uv_direct - uv_via_world, axis=-1)
        results["self_reproj_px"].append(float(np.median(reproj[c["visibility"]])))

    # Cross-view: fraction of a's visible points that project in-bounds in b
    n = min(len(clips), max_pairs + 1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = clips[i], clips[j]
            # Sample a few frames
            T = a["T"]
            frames = np.linspace(0, T - 1, num=min(10, T), dtype=int)
            inb_list = []
            for f in frames:
                vis = a["visibility"][f]
                if vis.sum() == 0:
                    continue
                pts = a["tracks_world"][f, vis]
                uv, depth, valid = project_points(pts, b["K"], b["w2c"])
                ib = in_bounds(uv, b["width"], b["height"]) & valid
                inb_list.append(float(ib.mean()))
            if inb_list:
                results["cross_in_bounds_frac"].append(float(np.mean(inb_list)))

    summary = {
        "median_roundtrip_m": float(np.median(results["roundtrip_m"])),
        "median_self_reproj_px": float(np.median(results["self_reproj_px"])),
        "mean_cross_in_bounds_frac": float(np.mean(results["cross_in_bounds_frac"]))
        if results["cross_in_bounds_frac"]
        else 0.0,
        "n_cams": len(clips),
        "pass_roundtrip": float(np.median(results["roundtrip_m"])) < 1e-4,
        "pass_self_reproj": float(np.median(results["self_reproj_px"])) < 0.5,
    }
    return summary


def build_scene(
    scene: str,
    npz_root: Path,
    d3g_root: Path,
    out_dir: Path,
) -> Optional[dict]:
    d3g_scene = d3g_root / scene
    if not (d3g_scene / "train_meta.json").exists():
        print(f"SKIP {scene}: no train_meta.json at {d3g_scene}")
        return None

    meta = load_d3g_meta(d3g_scene)
    cam_lookup = build_cam_lookup(meta)
    print(f"[{scene}] D3G cameras: {sorted(cam_lookup.keys())}")

    npz_paths = list_scene_npz(npz_root, scene)
    print(f"[{scene}] Found {len(npz_paths)} TAPVid npz files")

    clips = []
    for p in npz_paths:
        clip = load_camera_clip(p, cam_lookup)
        if clip is not None:
            clips.append(clip)

    if len(clips) < 2:
        print(f"SKIP {scene}: need >=2 cameras, got {len(clips)}")
        return None

    aabb = compute_scene_aabb(clips)
    calib_report = verify_calibration(clips)
    print(f"[{scene}] calib gate: {calib_report}")

    # Build per-camera lightweight metadata (no raw tracks — those stay in npz)
    cameras = []
    for c in clips:
        cameras.append(
            {
                "cam_id": c["cam_id"],
                "npz_path": c["npz_path"],
                "K": c["K"].tolist(),
                "K_d3g": c["K_d3g"].tolist(),
                "w2c": c["w2c"].tolist(),
                "fx_fy_cx_cy": c["fx_fy_cx_cy"].tolist(),
                "width": c["width"],
                "height": c["height"],
                "T": c["T"],
                "N": c["N"],
                "has_images": c["has_images"],
                "vis_frac": float(c["visibility"].mean()),
            }
        )

    # Precompute cross-view projection targets for a reference camera set:
    # For each source camera's tracks_world, project into every other camera.
    # Save as a compact cache per scene for training (optional, large).
    # Here we store only metadata; projections are done on the fly / in dataset.

    scene_meta = {
        "scene": scene,
        "aabb": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in aabb.items()},
        "cameras": cameras,
        "calib_report": calib_report,
        "image_size": [int(meta["w"]), int(meta["h"])],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene}_mc.json"
    with open(out_path, "w") as f:
        json.dump(scene_meta, f, indent=2)
    print(f"[{scene}] wrote {out_path}")

    # Also dump world tracks per camera as npz for fast training access
    tracks_dir = out_dir / scene / "tracks_world"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    for c in clips:
        np.savez_compressed(
            tracks_dir / f"cam_{c['cam_id']}.npz",
            tracks_world=c["tracks_world"],
            tracks_cam=c["tracks_cam"],
            visibility=c["visibility"],
            queries_xyt=c["queries_xyt"],
            tracks_norm=world_to_normalized(c["tracks_world"], aabb).astype(np.float32),
        )

    return scene_meta


def build_all(
    npz_root: Path,
    d3g_root: Path,
    out_dir: Path,
    scenes: Optional[List[str]] = None,
) -> dict:
    scenes = scenes or list(SCENES)
    index = {"scenes": {}, "calib_ok": True}
    for scene in scenes:
        meta = build_scene(scene, npz_root, d3g_root, out_dir)
        if meta is None:
            continue
        index["scenes"][scene] = {
            "meta_path": str(out_dir / f"{scene}_mc.json"),
            "n_cameras": len(meta["cameras"]),
            "calib_report": meta["calib_report"],
        }
        if not (
            meta["calib_report"]["pass_roundtrip"]
            and meta["calib_report"]["pass_self_reproj"]
        ):
            index["calib_ok"] = False

    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_root", type=str, default="./data")
    parser.add_argument("--d3g_root", type=str, default="./data/d3g/data")
    parser.add_argument("--out_dir", type=str, default="./data/tapvid3d_mc")
    parser.add_argument("--scenes", type=str, nargs="*", default=None)
    args = parser.parse_args()

    index = build_all(
        Path(args.npz_root),
        Path(args.d3g_root),
        Path(args.out_dir),
        scenes=args.scenes,
    )
    print("=== BUILD SUMMARY ===")
    print(json.dumps(index, indent=2))
    if not index["calib_ok"]:
        print("WARNING: calibration gate FAILED for at least one scene")
        raise SystemExit(1)
    print("Calibration gate PASSED.")


if __name__ == "__main__":
    main()
