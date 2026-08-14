"""Shared TAPVid-3D-MC evaluation protocol.

Used by ``evaluate_lapa.py`` and ``scripts/triangulation_baseline.py`` so a
known-good predictor (DLT) and the model are scored identically.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from lapa.eval.metrics import compute_tapvid3d_metrics


TAPVID_2D_THRESHOLDS = (1, 2, 4, 8, 16)


def w2c_from_normalized(
    w2c_n: np.ndarray,
    center: np.ndarray,
    half: np.ndarray,
) -> np.ndarray:
    """Invert ``build_w2c_normalized`` to recover a world-frame w2c."""
    Rn = np.asarray(w2c_n)[:3, :3]
    tn = np.asarray(w2c_n)[:3, 3]
    half = np.asarray(half).reshape(1, 3)
    center = np.asarray(center).reshape(3)
    R = Rn / half
    t = tn - R @ center
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c


def world_to_camera(points_world: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    return points_world @ R.T + t


def project_to_2d(points_cam: np.ndarray, fx_fy_cx_cy: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = fx_fy_cx_cy
    z = np.clip(points_cam[..., 2], 1e-6, None)
    u = fx * (points_cam[..., 0] / z) + cx
    v = fy * (points_cam[..., 1] / z) + cy
    return np.stack([u, v], axis=-1)


def compute_tapvid2d_metrics(
    gt_occluded: np.ndarray,
    gt_xy: np.ndarray,
    pred_occluded: np.ndarray,
    pred_xy: np.ndarray,
    image_size: Tuple[int, int],
    thresholds: Sequence[int] = TAPVID_2D_THRESHOLDS,
) -> Dict[str, float]:
    """TAP-Vid 2D AJ / APD on rasters rescaled to 256x256.

    Args:
        gt_occluded, pred_occluded: (N, T) True = occluded
        gt_xy, pred_xy: (N, T, 2) in original pixel coordinates
        image_size: (W, H) of the original frames
    """
    W, H = image_size
    scale = np.array([256.0 / max(W, 1), 256.0 / max(H, 1)], dtype=np.float64)
    gt = np.asarray(gt_xy, dtype=np.float64) * scale
    pred = np.asarray(pred_xy, dtype=np.float64) * scale
    gt_occ = np.asarray(gt_occluded, dtype=bool)
    pred_occ = np.asarray(pred_occluded, dtype=bool)
    visible = np.logical_not(gt_occ)
    pred_visible = np.logical_not(pred_occ)

    jaccards = []
    fracs = []
    dist2 = np.sum(np.square(pred - gt), axis=-1)
    for thresh in thresholds:
        within = dist2 < float(thresh) ** 2
        is_correct = np.logical_and(within, visible)
        denom_vis = max(float(visible.sum()), 1.0)
        fracs.append(float(is_correct.sum()) / denom_vis)

        true_pos = float(np.logical_and(is_correct, pred_visible).sum())
        gt_pos = float(visible.sum())
        false_pos = np.logical_or(
            np.logical_and(np.logical_not(visible), pred_visible),
            np.logical_and(np.logical_not(within), pred_visible),
        )
        jaccards.append(true_pos / max(gt_pos + float(false_pos.sum()), 1.0))

    return {
        "AJ2D": float(np.mean(jaccards)) * 100.0,
        "APD2D": float(np.mean(fracs)) * 100.0,
    }


def score_tracks(
    pred_world: np.ndarray,
    gt_world: np.ndarray,
    pred_visible: np.ndarray,
    gt_visible: np.ndarray,
    w2c_ref: np.ndarray,
    intrinsics: np.ndarray,
    image_size: Tuple[int, int],
) -> Dict[str, float]:
    """Score one sequence. Arrays are (T, M, 3) / (T, M).

    Returns paper-style percentages (0-100).
    """
    pred_cam = world_to_camera(pred_world, w2c_ref)
    gt_cam = world_to_camera(gt_world, w2c_ref)

    pred_tr = np.transpose(pred_cam, (1, 0, 2)).astype(np.float64)
    gt_tr = np.transpose(gt_cam, (1, 0, 2)).astype(np.float64)
    gt_occ = np.transpose(np.logical_not(gt_visible), (1, 0))
    pred_occ = np.transpose(np.logical_not(pred_visible), (1, 0))

    m3 = compute_tapvid3d_metrics(
        gt_occluded=gt_occ.astype(bool),
        gt_tracks=gt_tr,
        pred_occluded=pred_occ.astype(bool),
        pred_tracks=pred_tr,
        intrinsics_params=np.asarray(intrinsics, dtype=np.float64),
        scaling="median",
        order="n t",
    )

    pred_2d = project_to_2d(pred_cam, intrinsics)
    gt_2d = project_to_2d(gt_cam, intrinsics)
    m2 = compute_tapvid2d_metrics(
        gt_occluded=gt_occ,
        gt_xy=np.transpose(gt_2d, (1, 0, 2)),
        pred_occluded=pred_occ,
        pred_xy=np.transpose(pred_2d, (1, 0, 2)),
        image_size=image_size,
    )

    # Constant-visible predictor: always visible
    pred_occ_cv = np.zeros_like(gt_occ)
    m_cv = compute_tapvid3d_metrics(
        gt_occluded=gt_occ.astype(bool),
        gt_tracks=gt_tr,
        pred_occluded=pred_occ_cv,
        pred_tracks=pred_tr,
        intrinsics_params=np.asarray(intrinsics, dtype=np.float64),
        scaling="median",
        order="n t",
    )

    return {
        "APD": float(np.mean(m3["average_pts_within_thresh"])) * 100.0,
        "OA": float(np.mean(m3["occlusion_accuracy"])) * 100.0,
        "AJ3D": float(np.mean(m3["average_jaccard"])) * 100.0,
        "AJ2D": float(m2["AJ2D"]),
        "OA_const_vis": float(np.mean(m_cv["occlusion_accuracy"])) * 100.0,
        "vis_frac": float(gt_visible.mean()) * 100.0,
    }


def summarize(samples: Sequence[Dict]) -> Dict:
    if not samples:
        raise RuntimeError("No metrics computed")
    keys = ("APD", "OA", "AJ3D", "AJ2D", "OA_const_vis")
    out = {k: float(np.mean([s[k] for s in samples])) for k in keys}
    out["n_samples"] = len(samples)
    out["per_sample"] = list(samples)
    return out
