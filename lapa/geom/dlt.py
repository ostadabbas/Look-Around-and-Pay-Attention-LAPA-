"""Differentiable weighted DLT triangulation."""

from __future__ import annotations

from typing import List, Sequence, Union

import torch


def projection_matrices(
    view_K: Sequence[torch.Tensor],
    view_w2c: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Build 3x4 projection matrices P = K [R|t].

    Args:
        view_K: list of (3, 3)
        view_w2c: list of (4, 4) (world or normalized)
    Returns:
        P: (V, 3, 4)
    """
    ps = []
    for K, w2c in zip(view_K, view_w2c):
        ps.append(K @ w2c[:3, :4])
    return torch.stack(ps, dim=0)


def triangulate_dlt(
    uv: torch.Tensor,
    P: torch.Tensor,
    weights: torch.Tensor,
    min_views: int = 2,
    fallback: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    """Weighted DLT over V views for M points.

    Args:
        uv: (V, M, 2) pixel observations
        P: (V, 3, 4) projection matrices (same coordinate frame as output)
        weights: (V, M) non-negative observation weights
        min_views: require this many strictly-positive weights, else fallback
        fallback: (M, 3) used when a point has too few views (default zeros)
    Returns:
        points: (M, 3) inhomogeneous coordinates
    """
    V, M, _ = uv.shape
    device = uv.device
    dtype = uv.dtype
    w = weights.clamp(min=0).to(dtype)
    n_valid = (w > 1e-6).sum(dim=0)  # (M,)

    P0 = P[:, 0, :]  # (V, 4)
    P1 = P[:, 1, :]
    P2 = P[:, 2, :]
    u = uv[..., 0]
    v = uv[..., 1]
    # (V, M, 4)
    row0 = u.unsqueeze(-1) * P2.unsqueeze(1) - P0.unsqueeze(1)
    row1 = v.unsqueeze(-1) * P2.unsqueeze(1) - P1.unsqueeze(1)
    sw = w.sqrt().unsqueeze(-1)
    row0 = row0 * sw
    row1 = row1 * sw
    # (M, 2V, 4)
    A = torch.stack([row0, row1], dim=2).permute(1, 0, 2, 3).reshape(M, 2 * V, 4)

    # Guard against all-zero rows (no observations)
    mag = A.square().sum(dim=-1, keepdim=True).clamp(min=1e-12).sqrt()
    A = A / mag

    try:
        _, _, Vh = torch.linalg.svd(A, full_matrices=False)
    except RuntimeError:
        # CPU fallback for numerical issues
        _, _, Vh = torch.linalg.svd(A.cpu(), full_matrices=False)
        Vh = Vh.to(device=device, dtype=dtype)

    h = Vh[:, -1, :]  # (M, 4)
    homog = h[:, 3:4].clone()
    homog = torch.where(homog.abs() < 1e-8, torch.ones_like(homog), homog)
    pts = h[:, :3] / homog

    if fallback is None:
        fallback = torch.zeros(M, 3, device=device, dtype=dtype)
    else:
        fallback = fallback.to(device=device, dtype=dtype)
    pts = torch.where((n_valid >= min_views).unsqueeze(-1), pts, fallback)
    return pts


def _reproject(points: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Reproject (M,3) points with (V,3,4) matrices -> (V,M,2)."""
    ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
    homog = torch.cat([points, ones], dim=-1)  # (M, 4)
    proj = torch.einsum("vij,mj->vmi", P, homog)  # (V, M, 3)
    z = proj[..., 2].clamp(min=1e-6)
    return torch.stack([proj[..., 0] / z, proj[..., 1] / z], dim=-1)


def triangulate_dlt_irls(
    uv: torch.Tensor,
    P: torch.Tensor,
    weights: torch.Tensor,
    min_views: int = 2,
    fallback: Union[torch.Tensor, None] = None,
    iters: int = 2,
    sigma_px: float = 5.0,
) -> torch.Tensor:
    """Weighted DLT with a few IRLS reweighting steps on reprojection error.

    Lets geometrically-visible but drifting CoTracker observations be
    down-weighted instead of dominating the triangulation.
    """
    w = weights.clamp(min=0)
    pts = triangulate_dlt(uv, P, w, min_views=min_views, fallback=fallback)
    if iters <= 0:
        return pts
    soft = (w > 1e-6).to(uv.dtype)
    for _ in range(iters):
        uv_hat = _reproject(pts, P)
        err = (uv_hat - uv).norm(dim=-1)  # (V, M)
        rw = torch.exp(-(err / sigma_px).pow(2)) * soft
        # Keep a floor so a view does not vanish entirely in one step
        w = (0.05 * soft + 0.95 * rw) * weights.clamp(min=0)
        pts = triangulate_dlt(uv, P, w, min_views=min_views, fallback=fallback)
    return pts
