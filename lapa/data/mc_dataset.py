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
        use_gt_tracks: bool = True,
    ):
        self.mc_dir = Path(mc_dir)
        self.feature_dir = Path(feature_dir)
        self.num_views = num_views
        self.num_frames = num_frames
        self.max_points = max_points
        self.use_gt_tracks = use_gt_tracks
        self.fixed_sample: Optional[dict] = None  # set for overfit mode

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
        tracks_world = ref_tracks["tracks_world"]  # (T, N, 3)
        tracks_norm = ref_tracks["tracks_norm"]  # (T, N, 3)
        vis_ref = ref_tracks["visibility"]  # (T, N)
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

        aabb = meta["aabb"]
        center = torch.tensor(aabb["center"], dtype=torch.float32)
        half = torch.tensor(aabb["half"], dtype=torch.float32)

        # Reference GT
        gt_norm = torch.from_numpy(tracks_norm[frame_idx][:, point_idx]).float()  # (T, M, 3)
        gt_world = torch.from_numpy(tracks_world[frame_idx][:, point_idx]).float()
        visible = torch.from_numpy(vis_ref[frame_idx][:, point_idx]).bool()

        # Queries: first-frame GT positions (normalized)
        queries0 = gt_norm[0].clone()

        view_points_2d = []  # list[view] of list[t] of (K, 2) — for ref-aligned M points
        view_features = []
        view_K = []
        view_w2c_norm = []
        visible_per_view = []
        view_points_2d_native = []  # each view's own tracks (for attention)
        view_features_native = []
        image_size = (640, 360)

        for cam in chosen:
            K = torch.tensor(cam["K"], dtype=torch.float32)
            w2c = torch.tensor(cam["w2c"], dtype=torch.float32)
            w2c_n = build_w2c_normalized(w2c, center, half)
            view_K.append(K)
            view_w2c_norm.append(w2c_n)

            # Project reference world points into this camera for L_proj targets
            pts_w = gt_world.numpy()  # (T, M, 3)
            uv, depth, valid = project_points(pts_w, K.numpy(), w2c.numpy())
            ib = in_bounds(uv, cam["width"], cam["height"])
            vis_v = valid & ib & visible.numpy()
            visible_per_view.append(torch.from_numpy(vis_v).bool())

            # 2D targets for projection loss (ref identities)
            view_points_2d.append(torch.from_numpy(uv.astype(np.float32)))

            # Native tracks + features for volumetric attention
            with h5py.File(cam["feat_path"], "r") as f:
                key_2d = "tracks_2d_gt" if self.use_gt_tracks else "tracks_2d"
                tracks_2d = np.asarray(f[key_2d], dtype=np.float32)
                feats = np.asarray(f["features"], dtype=np.float16)
                vis_n = np.asarray(f["visibility"], dtype=bool)
                image_size = (int(f.attrs["W"]), int(f.attrs["H"]))

            Nn = tracks_2d.shape[1]
            # Subsample native points
            Kn = min(self.max_points, Nn)
            # Prefer visible in window start
            vis_start = vis_n[frame_idx[0]]
            cand = np.where(vis_start)[0]
            if len(cand) == 0:
                cand = np.arange(Nn)
            if len(cand) >= Kn:
                sel = np.array(rng.sample(list(cand), Kn), dtype=np.int64)
            else:
                sel = cand
            sel.sort()

            pts_list = []
            feat_list = []
            for t in frame_idx:
                pts_list.append(torch.from_numpy(tracks_2d[t, sel]).float())
                feat_list.append(torch.from_numpy(feats[t, sel].astype(np.float32)))
            view_points_2d_native.append(pts_list)
            view_features_native.append(feat_list)

        return {
            "scene": scene,
            "cam_ids": [c["cam_id"] for c in chosen],
            "gt_norm": gt_norm,  # (T, M, 3)
            "gt_world": gt_world,
            "visible": visible,  # (T, M)
            "queries0": queries0,  # (M, 3)
            "view_points_2d": view_points_2d,  # list[V] of (T, M, 2) — proj targets
            "visible_per_view": visible_per_view,
            "view_points_2d_native": view_points_2d_native,  # list[V][T] (K,2)
            "view_features_native": view_features_native,  # list[V][T] (K,768)
            "view_K": view_K,
            "view_w2c_norm": view_w2c_norm,
            "aabb_center": center,
            "aabb_half": half,
            "image_size": image_size,
        }


def collate_identity(batch):
    """Batch size 1 collate — return the single sample."""
    assert len(batch) == 1
    return batch[0]
