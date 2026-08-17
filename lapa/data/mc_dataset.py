"""Multi-camera dataset for LAPA training on TAPVid-3D-MC.

Yields random 3-camera triplets and temporal windows with:
  - per-view 2D tracks + DINOv2 features (from feature cache)
  - reference-view GT 3D tracks in normalized coords
  - camera K / w2c_norm
  - per-view visibility of reference points (geometric)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from lapa.data.mc_builder import (
    in_bounds,
    project_points,
    world_to_normalized,
)
from lapa.models.lapa import build_w2c_normalized


def load_split_files(data_root: Path) -> Tuple[set, set]:
    minival = set()
    full_eval = set()
    if (data_root / "minival_pstudio.txt").exists():
        minival = set((data_root / "minival_pstudio.txt").read_text().split())
    if (data_root / "full_eval_pstudio.txt").exists():
        full_eval = set((data_root / "full_eval_pstudio.txt").read_text().split())
    return minival, full_eval


class TAPVid3DMCDataset(Dataset):
    def __init__(
        self,
        mc_dir: str = "./data/tapvid3d_mc",
        feature_dir: str = "./data/feature_cache",
        data_root: str = "./data",
        split: str = "train",  # train=full_eval cams, val=minival cams
        num_views: int = 3,
        num_frames: int = 24,
        max_points: int = 64,
        scenes: Optional[Sequence[str]] = None,
        use_gt_tracks: bool = False,
    ):
        self.mc_dir = Path(mc_dir)
        self.feature_dir = Path(feature_dir)
        self.num_views = num_views
        self.num_frames = num_frames
        self.max_points = max_points
        self.use_gt_tracks = use_gt_tracks
        self.fixed_sample: Optional[dict] = None  # set for overfit mode
        self.canonical_feature_dir = Path(
            str(self.feature_dir).replace("feature_cache", "feature_cache_canonical")
            if "feature_cache_canonical" not in str(self.feature_dir)
            else self.feature_dir
        )
        # Keep original sparse cache for fallback appearance features
        self.sparse_feature_dir = Path(feature_dir)

        index = json.loads((self.mc_dir / "index.json").read_text())
        self.scenes = list(scenes) if scenes else list(index["scenes"].keys())
        minival, full_eval = load_split_files(Path(data_root))

        # Per scene: list of camera dicts filtered by split
        self.scene_cams: Dict[str, List[dict]] = {}
        self.scene_meta: Dict[str, dict] = {}
        for scene in self.scenes:
            meta = json.loads((self.mc_dir / f"{scene}_mc.json").read_text())
            self.scene_meta[scene] = meta
            cams = []
            for cam in meta["cameras"]:
                fname = Path(cam["npz_path"]).name
                feat_path = self.feature_dir / scene / f"cam_{cam['cam_id']}.h5"
                if not feat_path.exists():
                    continue
                if split == "train" and full_eval and fname not in full_eval:
                    continue
                if split in ("val", "minival") and minival and fname not in minival:
                    continue
                cam = dict(cam)
                cam["feat_path"] = str(feat_path)
                cam["tracks_path"] = str(
                    self.mc_dir / scene / "tracks_world" / f"cam_{cam['cam_id']}.npz"
                )
                cams.append(cam)
            if len(cams) >= num_views:
                self.scene_cams[scene] = cams

        self.scene_list = [s for s in self.scenes if s in self.scene_cams]
        if not self.scene_list:
            raise RuntimeError(
                f"No scenes with >= {num_views} cameras for split={split}. "
                f"Did you run feature precompute?"
            )

        # Virtual length
        self.length = 2000 if split == "train" else 200

    def __len__(self) -> int:
        return self.length

    def _load_cam(self, scene: str, cam: dict, frame_idx: np.ndarray, point_idx: np.ndarray):
        with h5py.File(cam["feat_path"], "r") as f:
            key_2d = "tracks_2d_gt" if self.use_gt_tracks else "tracks_2d"
            tracks_2d = np.asarray(f[key_2d], dtype=np.float32)  # (T, N, 2)
            feats = np.asarray(f["features"], dtype=np.float16)  # (T, N, 768)
            vis = np.asarray(f["visibility"], dtype=bool)
            W = int(f.attrs["W"])
            H = int(f.attrs["H"])

        # Subsample points that exist in this camera
        N = tracks_2d.shape[1]
        # point_idx may be for reference camera; for non-ref we use own points
        return tracks_2d, feats, vis, W, H, N

    def __getitem__(self, index: int) -> dict:
        if self.fixed_sample is not None:
            # Return a deep-enough copy of tensors so training doesn't alias
            sample = self.fixed_sample
            return {
                k: (v.clone() if torch.is_tensor(v) else
                    [x.clone() if torch.is_tensor(x) else
                     [y.clone() if torch.is_tensor(y) else y for y in x]
                     if isinstance(x, list) else x
                     for x in v]
                    if isinstance(v, list) else v)
                for k, v in sample.items()
            }
        rng = random.Random(index * 10007 + 13)
        scene = rng.choice(self.scene_list)
        meta = self.scene_meta[scene]
        cams = self.scene_cams[scene]
        chosen = rng.sample(cams, self.num_views)
        ref = chosen[0]

        # Load reference world tracks
        ref_tracks = np.load(ref["tracks_path"])
        canon_path = self.mc_dir / scene / "canonical.npz"
        canonical_idx = None
        if canon_path.exists():
            canon = np.load(canon_path)
            tracks_world = np.asarray(canon["tracks_world"], dtype=np.float32)
            tracks_norm = np.asarray(canon["tracks_norm"], dtype=np.float32)
            vis_ref = np.asarray(canon["visibility"], dtype=bool)
        else:
            tracks_world = ref_tracks["tracks_world"]
            tracks_norm = ref_tracks["tracks_norm"]
            vis_ref = ref_tracks["visibility"]
        T_full, N_full, _ = tracks_world.shape

        # Temporal window
        if T_full <= self.num_frames:
            frame_idx = np.arange(T_full)
        else:
            start = rng.randint(0, T_full - self.num_frames)
            frame_idx = np.arange(start, start + self.num_frames)

        # Point subsample: prefer points visible in first frame of window
        vis0 = vis_ref[frame_idx[0]]
        candidates = np.where(vis0)[0]
        if len(candidates) == 0:
            candidates = np.arange(N_full)
        M = min(self.max_points, len(candidates))
        point_idx = np.array(rng.sample(list(candidates), M), dtype=np.int64)
        point_idx.sort()
        if canon_path.exists():
            canonical_idx = point_idx

        aabb = meta["aabb"]
        center = torch.tensor(aabb["center"], dtype=torch.float32)
        half = torch.tensor(aabb["half"], dtype=torch.float32)

        # Reference GT
        gt_norm = torch.from_numpy(tracks_norm[frame_idx][:, point_idx]).float()  # (T, M, 3)
        gt_world = torch.from_numpy(tracks_world[frame_idx][:, point_idx]).float()
        visible = torch.from_numpy(vis_ref[frame_idx][:, point_idx]).bool()

        queries0 = gt_norm[0].clone()

        view_points_2d = []
        view_K = []
        view_w2c_norm = []
        visible_per_view = []
        view_points_2d_native = []
        view_features_native = []
        image_size = (640, 360)

        for cam in chosen:
            packed = load_corresponded_view(
                cam=cam,
                scene=scene,
                frame_idx=frame_idx,
                point_idx=point_idx,
                gt_world=gt_world.numpy(),
                visible_ref=visible.numpy(),
                center=center,
                half=half,
                use_gt_tracks=self.use_gt_tracks,
                canonical_feature_dir=self.canonical_feature_dir,
                sparse_feat_path=cam["feat_path"],
                canonical_idx=canonical_idx,
            )
            view_K.append(packed["K"])
            view_w2c_norm.append(packed["w2c_norm"])
            visible_per_view.append(packed["valid"])
            view_points_2d.append(packed["uv_gt"])
            view_points_2d_native.append(packed["pts_list"])
            view_features_native.append(packed["feat_list"])
            image_size = packed["image_size"]

        return {
            "scene": scene,
            "cam_ids": [c["cam_id"] for c in chosen],
            "gt_norm": gt_norm,
            "gt_world": gt_world,
            "visible": visible,
            "queries0": queries0,
            "view_points_2d": view_points_2d,
            "visible_per_view": visible_per_view,
            "view_points_2d_native": view_points_2d_native,
            "view_features_native": view_features_native,
            "view_K": view_K,
            "view_w2c_norm": view_w2c_norm,
            "view_w2c_world": [
                torch.tensor(c["w2c"], dtype=torch.float32) for c in chosen
            ],
            "aabb_center": center,
            "aabb_half": half,
            "image_size": image_size,
        }


def camera_center(w2c: np.ndarray) -> np.ndarray:
    R, t = np.asarray(w2c)[:3, :3], np.asarray(w2c)[:3, 3]
    return (-R.T @ t).astype(np.float64)


def pick_companion_cameras(ref: dict, cams: List[dict], k: int = 2) -> List[dict]:
    """Pick k cameras with the largest baseline from the reference."""
    rc = camera_center(ref["w2c"])
    others = [c for c in cams if int(c["cam_id"]) != int(ref["cam_id"])]
    others.sort(
        key=lambda c: -float(np.linalg.norm(camera_center(c["w2c"]) - rc))
    )
    if len(others) < k:
        raise RuntimeError(
            f"Need {k} companion cameras for cam {ref['cam_id']}, got {len(others)}"
        )
    return others[:k]


def _nearest_feature(native_uv: np.ndarray, native_feat: np.ndarray, query_uv: np.ndarray) -> np.ndarray:
    """Nearest-neighbour appearance lookup. native_uv (K,2), query_uv (M,2)."""
    if native_uv.shape[0] == 0:
        return np.zeros((query_uv.shape[0], native_feat.shape[-1] if native_feat.ndim == 2 else 768), dtype=np.float32)
    d = (
        (query_uv[:, None, 0] - native_uv[None, :, 0]) ** 2
        + (query_uv[:, None, 1] - native_uv[None, :, 1]) ** 2
    )
    nn = d.argmin(axis=1)
    return native_feat[nn].astype(np.float32)


def load_corresponded_view(
    cam: dict,
    scene: str,
    frame_idx: np.ndarray,
    point_idx: np.ndarray,
    gt_world: np.ndarray,
    visible_ref: np.ndarray,
    center: torch.Tensor,
    half: torch.Tensor,
    use_gt_tracks: bool,
    canonical_feature_dir: Path,
    sparse_feat_path: str,
    canonical_idx: Optional[np.ndarray] = None,
    load_features: bool = True,
) -> dict:
    """Load per-view 2D observations for the SAME identities as ``gt_world``.

    Prefer a canonical CoTracker cache (same index space across cameras).
    Fall back to GT projections + nearest-neighbour features from the sparse cache.

    Query-frame (t=0 of the window) 2D is always the GT projection.
    """
    K = torch.tensor(cam["K"], dtype=torch.float32)
    w2c = torch.tensor(cam["w2c"], dtype=torch.float32)
    w2c_n = build_w2c_normalized(w2c, center, half)
    uv_gt, depth, valid_depth = project_points(gt_world, K.numpy(), w2c.numpy())
    ib = in_bounds(uv_gt, cam["width"], cam["height"])
    # Geometric visibility in THIS camera (do not require ref-cam visibility:
    # companions must still contribute observations for triangulation).
    valid_geom = valid_depth & ib
    valid = valid_geom & visible_ref
    uv_gt = uv_gt.astype(np.float32)

    image_size = (int(cam["width"]), int(cam["height"]))
    uv_obs = uv_gt.copy()
    feats_all = None  # (T_window, M, 768)

    canon_h5 = Path(canonical_feature_dir) / scene / f"cam_{cam['cam_id']}.h5"
    if canon_h5.exists() and canonical_idx is not None:
        with h5py.File(canon_h5, "r") as f:
            key = "tracks_2d_gt" if use_gt_tracks else "tracks_2d"
            t0 = int(frame_idx[0])
            t1 = int(frame_idx[-1]) + 1
            # Contiguous window read then column-subsample (fast path)
            tracks_win = np.asarray(f[key][t0:t1], dtype=np.float32)
            tracks_gt_win = np.asarray(f["tracks_2d_gt"][t0:t1], dtype=np.float32)
            tracks = tracks_win[:, canonical_idx]
            tracks_gt0 = tracks_gt_win[0, canonical_idx]
            if load_features:
                feats_all = np.asarray(f["features"][t0:t1], dtype=np.float32)[
                    :, canonical_idx
                ]
            else:
                feats_all = np.zeros((len(frame_idx), len(canonical_idx), 768), dtype=np.float32)
            # GT / CoTracker visibility from the cache.
            if use_gt_tracks:
                vis_key = "visibility"
            else:
                vis_key = (
                    "visibility_tracker" if "visibility_tracker" in f else "visibility"
                )
            vis_obs = np.asarray(f[vis_key][t0:t1], dtype=bool)[:, canonical_idx]
            image_size = (
                int(f.attrs.get("W", cam["width"])),
                int(f.attrs.get("H", cam["height"])),
            )
        uv_obs = tracks.copy()
        uv_obs[0] = tracks_gt0  # query frame = GT
        if use_gt_tracks:
            valid = valid_geom & visible_ref
        else:
            # Honest CoTracker observations only (geom ∩ tracker).
            valid = valid_geom & vis_obs
            # Always trust the query frame GT projection when geometrically valid.
            valid[0] = valid_geom[0] & visible_ref[0]

    if feats_all is None and load_features:
        # Sparse cache nearest-neighbour appearance at GT 2D
        feats_all = np.zeros((len(frame_idx), gt_world.shape[1], 768), dtype=np.float32)
        if Path(sparse_feat_path).exists():
            with h5py.File(sparse_feat_path, "r") as f:
                native_uv = np.asarray(f["tracks_2d_gt"], dtype=np.float32)
                native_feat = np.asarray(f["features"], dtype=np.float32)
                image_size = (int(f.attrs.get("W", image_size[0])), int(f.attrs.get("H", image_size[1])))
            for i, t in enumerate(frame_idx):
                feats_all[i] = _nearest_feature(native_uv[t], native_feat[t], uv_gt[i])

        if not use_gt_tracks:
            pass

    if feats_all is None:
        feats_all = np.zeros((len(frame_idx), gt_world.shape[1], 768), dtype=np.float32)

    pts_list = [torch.from_numpy(uv_obs[i]).float() for i in range(len(frame_idx))]
    feat_list = [torch.from_numpy(feats_all[i]).float() for i in range(len(frame_idx))]
    return {
        "K": K,
        "w2c_norm": w2c_n,
        "uv_gt": torch.from_numpy(uv_gt.astype(np.float32)),
        "valid": torch.from_numpy(valid.astype(bool)),
        "pts_list": pts_list,
        "feat_list": feat_list,
        "image_size": image_size,
    }


class TAPVid3DMCEvalDataset(Dataset):
    """One full-sequence sample per minival camera (reference) + 2 companions.

    GT 3D is the official TAPVid-3D tracks of the reference camera.
    Per-view 2D observations are CoTracker (if cached) with the query frame
    forced to the GT projection; otherwise GT projections.
    """

    def __init__(
        self,
        mc_dir: str = "./data/tapvid3d_mc",
        feature_dir: str = "./data/feature_cache",
        data_root: str = "./data",
        num_views: int = 3,
        max_points: int = 256,
        use_gt_tracks: bool = False,
        canonical_feature_dir: str = "./data/feature_cache_canonical",
        eval_feature_dir: str = "./data/feature_cache_eval",
        load_features: bool = True,
    ):
        self.mc_dir = Path(mc_dir)
        self.feature_dir = Path(feature_dir)
        self.canonical_feature_dir = Path(canonical_feature_dir)
        self.eval_feature_dir = Path(eval_feature_dir)
        self.num_views = num_views
        self.max_points = max_points
        self.use_gt_tracks = use_gt_tracks
        self.load_features = load_features

        minival, _ = load_split_files(Path(data_root))
        if not minival:
            raise RuntimeError(f"No minival_pstudio.txt under {data_root}")

        index = json.loads((self.mc_dir / "index.json").read_text())
        self.scene_meta: Dict[str, dict] = {}
        self.scene_cams: Dict[str, List[dict]] = {}
        for scene in index["scenes"]:
            meta = json.loads((self.mc_dir / f"{scene}_mc.json").read_text())
            self.scene_meta[scene] = meta
            cams = []
            for cam in meta["cameras"]:
                cam = dict(cam)
                cam["feat_path"] = str(self.feature_dir / scene / f"cam_{cam['cam_id']}.h5")
                cam["tracks_path"] = str(
                    self.mc_dir / scene / "tracks_world" / f"cam_{cam['cam_id']}.npz"
                )
                cams.append(cam)
            self.scene_cams[scene] = cams

        samples = []
        for fname in sorted(minival):
            if not fname.endswith(".npz"):
                continue
            stem = fname.replace(".npz", "")
            scene, cam_str = stem.split("_")
            cam_id = int(cam_str)
            if scene not in self.scene_cams:
                continue
            ref = next((c for c in self.scene_cams[scene] if int(c["cam_id"]) == cam_id), None)
            if ref is None:
                continue
            try:
                companions = pick_companion_cameras(ref, self.scene_cams[scene], k=num_views - 1)
            except RuntimeError:
                continue
            samples.append({"scene": scene, "ref": ref, "companions": companions, "fname": fname})
        if not samples:
            raise RuntimeError("No minival eval samples could be constructed")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        spec = self.samples[index]
        scene = spec["scene"]
        ref = spec["ref"]
        chosen = [ref] + spec["companions"]
        meta = self.scene_meta[scene]

        tracks = np.load(ref["tracks_path"])
        tracks_world = np.asarray(tracks["tracks_world"], dtype=np.float32)
        tracks_norm = np.asarray(tracks["tracks_norm"], dtype=np.float32)
        vis_ref = np.asarray(tracks["visibility"], dtype=bool)
        T_full, N_full, _ = tracks_world.shape
        frame_idx = np.arange(T_full)

        vis0 = vis_ref[0]
        candidates = np.where(vis0)[0]
        if len(candidates) == 0:
            candidates = np.arange(N_full)
        M = min(self.max_points, len(candidates))
        point_idx = candidates[:M].astype(np.int64)

        aabb = meta["aabb"]
        center = torch.tensor(aabb["center"], dtype=torch.float32)
        half = torch.tensor(aabb["half"], dtype=torch.float32)

        gt_norm = torch.from_numpy(tracks_norm[frame_idx][:, point_idx]).float()
        gt_world = torch.from_numpy(tracks_world[frame_idx][:, point_idx]).float()
        visible = torch.from_numpy(vis_ref[frame_idx][:, point_idx]).bool()
        queries0 = gt_norm[0].clone()

        view_points_2d = []
        view_K = []
        view_w2c_norm = []
        visible_per_view = []
        view_points_2d_native = []
        view_features_native = []
        image_size = (int(ref["width"]), int(ref["height"]))

        for cam in chosen:
            eval_h5 = (
                self.eval_feature_dir
                / scene
                / f"ref{ref['cam_id']}_cam{cam['cam_id']}.h5"
            )
            packed = load_corresponded_view(
                cam=cam,
                scene=scene,
                frame_idx=frame_idx,
                point_idx=point_idx,
                gt_world=gt_world.numpy(),
                visible_ref=visible.numpy(),
                center=center,
                half=half,
                use_gt_tracks=self.use_gt_tracks,
                canonical_feature_dir=eval_h5.parent if eval_h5.exists() else self.canonical_feature_dir,
                sparse_feat_path=cam["feat_path"],
                load_features=self.load_features,
            )
            if eval_h5.exists() and not self.use_gt_tracks:
                with h5py.File(eval_h5, "r") as f:
                    tracks = np.asarray(f["tracks_2d"], dtype=np.float32)
                    tracks_gt = np.asarray(f["tracks_2d_gt"], dtype=np.float32)
                    feats = np.asarray(f["features"], dtype=np.float32)
                    vis_tr = np.asarray(f["visibility_tracker"], dtype=bool)
                    vis_geom = np.asarray(f["visibility"], dtype=bool)
                n = min(tracks.shape[1], gt_world.shape[1])
                uv_obs = tracks[:, :n].copy()
                uv_obs[0] = tracks_gt[0, :n]
                packed["pts_list"] = [
                    torch.from_numpy(uv_obs[t]).float() for t in range(uv_obs.shape[0])
                ]
                packed["feat_list"] = [
                    torch.from_numpy(feats[t, :n].astype(np.float32))
                    for t in range(feats.shape[0])
                ]
                valid = vis_geom[:, :n] & vis_tr[:, :n]
                # Query frame: GT projection is trusted whenever geometric.
                valid[0] = vis_geom[0, :n]
                packed["valid"] = torch.from_numpy(valid)
            elif eval_h5.exists() and self.use_gt_tracks:
                with h5py.File(eval_h5, "r") as f:
                    tracks_gt = np.asarray(f["tracks_2d_gt"], dtype=np.float32)
                    vis_geom = np.asarray(f["visibility"], dtype=bool)
                    if self.load_features:
                        feats = np.asarray(f["features"], dtype=np.float32)
                    else:
                        feats = None
                n = min(tracks_gt.shape[1], gt_world.shape[1])
                packed["pts_list"] = [
                    torch.from_numpy(tracks_gt[t, :n]).float()
                    for t in range(tracks_gt.shape[0])
                ]
                if feats is not None:
                    packed["feat_list"] = [
                        torch.from_numpy(feats[t, :n].astype(np.float32))
                        for t in range(feats.shape[0])
                    ]
                packed["valid"] = torch.from_numpy(vis_geom[:, :n]) & visible
            view_K.append(packed["K"])
            view_w2c_norm.append(packed["w2c_norm"])
            visible_per_view.append(packed["valid"])
            view_points_2d.append(packed["uv_gt"])
            view_points_2d_native.append(packed["pts_list"])
            view_features_native.append(packed["feat_list"])
            image_size = packed["image_size"]

        return {
            "scene": scene,
            "cam_ids": [c["cam_id"] for c in chosen],
            "fname": spec["fname"],
            "gt_norm": gt_norm,
            "gt_world": gt_world,
            "visible": visible,
            "queries0": queries0,
            "view_points_2d": view_points_2d,
            "visible_per_view": visible_per_view,
            "view_points_2d_native": view_points_2d_native,
            "view_features_native": view_features_native,
            "view_K": view_K,
            "view_w2c_norm": view_w2c_norm,
            "view_w2c_world": [
                torch.tensor(c["w2c"], dtype=torch.float32) for c in chosen
            ],
            "aabb_center": center,
            "aabb_half": half,
            "image_size": image_size,
        }


def collate_identity(batch):
    """Batch size 1 collate — return the single sample."""
    assert len(batch) == 1
    return batch[0]
