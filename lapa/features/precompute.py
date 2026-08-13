"""Precompute DINOv2 patch features and 2D tracks for TAPVid-3D-MC clips.

For each (scene, camera) npz:
  - Decode JPEG frames
  - Project GT 3D tracks to 2D (oracle tracks; CoTracker optional)
  - Extract DINOv2 ViT-B/14 patch features at track locations
  - Save fp16 HDF5 cache

CoTracker can be enabled with --use_cotracker (slower; used for inference fidelity).
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def decode_images(images_jpeg_bytes) -> np.ndarray:
    """Decode to (T, H, W, 3) uint8 RGB."""
    frames = []
    for b in images_jpeg_bytes:
        if isinstance(b, np.ndarray):
            b = b.tobytes() if b.dtype != object else bytes(b)
        elif not isinstance(b, (bytes, bytearray)):
            b = bytes(b)
        img = Image.open(io.BytesIO(b)).convert("RGB")
        frames.append(np.asarray(img))
    return np.stack(frames, axis=0)


def project_cam_tracks_to_2d(tracks_cam: np.ndarray, fx_fy_cx_cy: np.ndarray) -> np.ndarray:
    """Project camera-frame 3D tracks to pixels. tracks_cam: (T, N, 3) -> (T, N, 2)."""
    fx, fy, cx, cy = fx_fy_cx_cy
    z = np.clip(tracks_cam[..., 2], 1e-6, None)
    u = fx * (tracks_cam[..., 0] / z) + cx
    v = fy * (tracks_cam[..., 1] / z) + cy
    return np.stack([u, v], axis=-1).astype(np.float32)


class DINOv2FeatureExtractor:
    def __init__(self, device: torch.device, image_size: int = 518):
        from transformers import AutoModel

        self.device = device
        self.image_size = image_size  # DINOv2 preferred multiple of 14
        # Make divisible by 14
        self.image_size = (image_size // 14) * 14
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        self.patch_size = 14
        self.embed_dim = 768

    @torch.no_grad()
    def extract_video_features(self, frames_rgb: np.ndarray) -> torch.Tensor:
        """frames_rgb: (T, H, W, 3) uint8 -> patch tokens (T, Gh, Gw, 768) float16."""
        T, H, W, _ = frames_rgb.shape
        # Process in batches to fit memory
        tokens = []
        bs = 8
        for start in range(0, T, bs):
            batch = frames_rgb[start : start + bs]
            x = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
            # ImageNet normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x = (x - mean) / std
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).to(self.device)
            out = self.model(pixel_values=x)
            # last_hidden_state: (B, 1+Gh*Gw, 768) — skip CLS
            hid = out.last_hidden_state[:, 1:, :]
            Gh = Gw = self.image_size // self.patch_size
            hid = hid.reshape(-1, Gh, Gw, self.embed_dim)
            tokens.append(hid.half().cpu())
        return torch.cat(tokens, dim=0)  # (T, Gh, Gw, 768)

    def sample_at_points(
        self,
        patch_tokens: torch.Tensor,
        points_2d: np.ndarray,
        orig_size: Tuple[int, int],
    ) -> np.ndarray:
        """Sample patch features at 2D points.

        patch_tokens: (T, Gh, Gw, 768)
        points_2d: (T, N, 2) in original pixel coords
        orig_size: (W, H)
        Returns: (T, N, 768) float16 numpy
        """
        T, Gh, Gw, D = patch_tokens.shape
        W, H = orig_size
        # Map pixels to patch indices
        # After resize to image_size, pixel (u,v) -> (u/W * image_size, v/H * image_size)
        # patch idx = floor(coord / patch_size)
        u = points_2d[..., 0] / W * self.image_size
        v = points_2d[..., 1] / H * self.image_size
        gi = np.clip(np.floor(u / self.patch_size).astype(np.int64), 0, Gw - 1)
        gj = np.clip(np.floor(v / self.patch_size).astype(np.int64), 0, Gh - 1)
        # Index
        t_idx = np.arange(T)[:, None]
        feats = patch_tokens.numpy()[t_idx, gj, gi]  # (T, N, 768)
        return feats.astype(np.float16)


def run_cotracker(
    frames_rgb: np.ndarray,
    queries_xyt: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run CoTracker3 offline.

    frames_rgb: (T, H, W, 3)
    queries_xyt: (N, 3) with (x, y, t)
    Returns: tracks (T, N, 2), visibility (T, N)
    """
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(device).eval()
    video = (
        torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).float()[None].to(device)
    )  # (1, T, 3, H, W)
    # CoTracker queries: (1, N, 3) as (t, x, y)
    q = torch.zeros(1, queries_xyt.shape[0], 3, device=device)
    q[0, :, 0] = torch.from_numpy(queries_xyt[:, 2]).float().to(device)  # t
    q[0, :, 1] = torch.from_numpy(queries_xyt[:, 0]).float().to(device)  # x
    q[0, :, 2] = torch.from_numpy(queries_xyt[:, 1]).float().to(device)  # y
    with torch.no_grad():
        pred_tracks, pred_visibility = model(video, queries=q)
    # pred_tracks: (1, T, N, 2), pred_visibility: (1, T, N)
    tracks = pred_tracks[0].cpu().numpy().astype(np.float32)
    vis = pred_visibility[0].cpu().numpy()
    if vis.dtype != np.bool_:
        vis = vis > 0.5
    return tracks, vis.astype(bool)


def process_clip(
    npz_path: Path,
    out_h5: Path,
    dino: DINOv2FeatureExtractor,
    use_cotracker: bool,
    device: torch.device,
    max_points: Optional[int] = None,
) -> None:
    if out_h5.exists() and out_h5.stat().st_size > 1000:
        return

    data = np.load(npz_path, allow_pickle=True)
    frames = decode_images(data["images_jpeg_bytes"])
    T, H, W, _ = frames.shape
    tracks_cam = np.asarray(data["tracks_XYZ"], dtype=np.float32)
    visibility = np.asarray(data["visibility"], dtype=bool)
    fx_fy_cx_cy = np.asarray(data["fx_fy_cx_cy"], dtype=np.float64)
    queries_xyt = np.asarray(data["queries_xyt"], dtype=np.float64)

    N = tracks_cam.shape[1]
    if max_points is not None and N > max_points:
        rng = np.random.RandomState(0)
        idx = rng.choice(N, max_points, replace=False)
        idx.sort()
        tracks_cam = tracks_cam[:, idx]
        visibility = visibility[:, idx]
        queries_xyt = queries_xyt[idx]
        N = max_points

    tracks_2d_gt = project_cam_tracks_to_2d(tracks_cam, fx_fy_cx_cy)

    if use_cotracker:
        tracks_2d, vis_ct = run_cotracker(frames, queries_xyt, device)
        # Prefer CoTracker where available; fall back to GT if NaN
        bad = ~np.isfinite(tracks_2d).all(axis=-1)
        tracks_2d[bad] = tracks_2d_gt[bad]
        visibility_ct = vis_ct
    else:
        tracks_2d = tracks_2d_gt
        visibility_ct = visibility

    # DINOv2 features at track locations (use GT 2D for stable sampling)
    patch_tokens = dino.extract_video_features(frames)
    feats = dino.sample_at_points(patch_tokens, tracks_2d_gt, (W, H))

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_h5.with_suffix(".partial.h5")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("tracks_2d", data=tracks_2d, compression="gzip")
        f.create_dataset("tracks_2d_gt", data=tracks_2d_gt, compression="gzip")
        f.create_dataset("visibility", data=visibility.astype(np.uint8), compression="gzip")
        f.create_dataset("visibility_tracker", data=visibility_ct.astype(np.uint8), compression="gzip")
        f.create_dataset("features", data=feats, compression="gzip")
        f.create_dataset("queries_xyt", data=queries_xyt.astype(np.float32))
        f.create_dataset("fx_fy_cx_cy", data=fx_fy_cx_cy.astype(np.float32))
        f.attrs["T"] = T
        f.attrs["N"] = N
        f.attrs["H"] = H
        f.attrs["W"] = W
        f.attrs["dino_dim"] = 768
        f.attrs["use_cotracker"] = int(use_cotracker)
        f.attrs["npz_path"] = str(npz_path)
    tmp.replace(out_h5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc_dir", type=str, default="./data/tapvid3d_mc")
    parser.add_argument("--out_dir", type=str, default="./data/feature_cache")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--use_cotracker", action="store_true")
    parser.add_argument("--max_points", type=int, default=256)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--split", type=str, default="all", choices=["all", "minival", "full_eval"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    dino = DINOv2FeatureExtractor(device)

    mc_dir = Path(args.mc_dir)
    out_dir = Path(args.out_dir)
    index = json.loads((mc_dir / "index.json").read_text())
    scenes = args.scenes or list(index["scenes"].keys())

    # Split filter via npz path
    minival_files = set()
    full_eval_files = set()
    data_root = Path("./data")
    if (data_root / "minival_pstudio.txt").exists():
        minival_files = set((data_root / "minival_pstudio.txt").read_text().split())
    if (data_root / "full_eval_pstudio.txt").exists():
        full_eval_files = set((data_root / "full_eval_pstudio.txt").read_text().split())
    else:
        # rebuild from disk
        full_eval_files = {p.name for p in (data_root / "tapvid3d_full_eval").rglob("*.npz")}

    tasks = []
    for scene in scenes:
        meta = json.loads((mc_dir / f"{scene}_mc.json").read_text())
        for cam in meta["cameras"]:
            npz_path = Path(cam["npz_path"])
            fname = npz_path.name
            if args.split == "minival" and fname not in minival_files:
                continue
            if args.split == "full_eval" and fname not in full_eval_files:
                continue
            out_h5 = out_dir / scene / f"cam_{cam['cam_id']}.h5"
            tasks.append((npz_path, out_h5, cam["cam_id"]))

    print(f"Processing {len(tasks)} clips (cotracker={args.use_cotracker})")
    for npz_path, out_h5, cam_id in tqdm(tasks, desc="precompute"):
        try:
            process_clip(
                npz_path,
                out_h5,
                dino,
                use_cotracker=args.use_cotracker,
                device=device,
                max_points=args.max_points,
            )
        except Exception as e:
            print(f"ERROR {npz_path}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
