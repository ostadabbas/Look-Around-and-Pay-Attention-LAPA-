"""LAPA multi-objective loss: L = λ1 L_recon + λ2 L_proj + λ3 L_attn (+ vis)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LAPALoss(nn.Module):
    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_proj: float = 0.7,
        lambda_attn: float = 0.8,
        lambda_vis: float = 0.5,
        lambda_view: float = 1.0,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_proj = lambda_proj
        self.lambda_attn = lambda_attn
        self.lambda_vis = lambda_vis
        self.lambda_view = lambda_view

    def view_weight_loss(
        self,
        view_w: torch.Tensor,
        obs_2d: List[torch.Tensor],
        gt_norm: torch.Tensor,
        view_K: List[torch.Tensor],
        view_w2c_norm: List[torch.Tensor],
        visible_per_view: List[torch.Tensor],
        image_size: Tuple[int, int] = (640, 360),
    ) -> torch.Tensor:
        """Push view weights onto low-reprojection-error observations.

        view_w: (T, V, M)  obs_2d[a]: (T, M, 2) corresponded tracker/GT 2D.
        """
        T, V, M = view_w.shape
        W, H = image_size
        diag = float((W ** 2 + H ** 2) ** 0.5)
        errs = []
        valids = []
        for a in range(V):
            K = view_K[a]
            w2c = view_w2c_norm[a]
            R, t = w2c[:3, :3], w2c[:3, 3]
            pts_cam = torch.einsum("ij,tmj->tmi", R, gt_norm) + t
            z = pts_cam[..., 2].clamp(min=1e-6)
            u = K[0, 0] * (pts_cam[..., 0] / z) + K[0, 2]
            v = K[1, 1] * (pts_cam[..., 1] / z) + K[1, 2]
            uv_gt = torch.stack([u, v], dim=-1)
            err = ((obs_2d[a] - uv_gt) / diag).pow(2).sum(dim=-1).sqrt()  # (T, M)
            errs.append(err)
            valids.append(visible_per_view[a].float())
        err = torch.stack(errs, dim=1)  # (T, V, M)
        valid = torch.stack(valids, dim=1)
        # Softmax over views so weights compete; invalid views get -inf.
        logits = torch.log(view_w.clamp(min=1e-6))
        logits = logits.masked_fill(valid < 0.5, -1e4)
        w = torch.softmax(logits, dim=1)
        return (w * err * valid).sum() / valid.sum().clamp(min=1.0)

    def reconstruction_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        visible: torch.Tensor,
        aabb_half: Optional[torch.Tensor] = None,
        aabb_center: Optional[torch.Tensor] = None,
        huber_delta: float = 0.05,
    ) -> torch.Tensor:
        """Huber reconstruction in metres (falls back to normalized L2).

        pred, gt: (T, M, 3); visible: (T, M)
        """
        if aabb_half is not None and aabb_center is not None:
            pred_m = pred * aabb_half + aabb_center
            gt_m = gt * aabb_half + aabb_center
        else:
            pred_m = pred
            gt_m = gt
        dist = (pred_m - gt_m).norm(dim=-1)
        delta = pred.new_tensor(huber_delta)
        huber = torch.where(
            dist < delta,
            0.5 * dist.square(),
            delta * (dist - 0.5 * delta),
        )
        w = visible.float()
        denom = w.sum().clamp(min=1.0)
        return (huber * w).sum() / denom

    def projection_loss(
        self,
        pred_norm: torch.Tensor,
        gt_points_2d: List[torch.Tensor],
        view_K: List[torch.Tensor],
        view_w2c_norm: List[torch.Tensor],
        visible_per_view: List[torch.Tensor],
        point_indices_per_view: Optional[List[torch.Tensor]] = None,
        image_size: Tuple[int, int] = (640, 360),
    ) -> torch.Tensor:
        """L_proj: reproject predicted 3D into each view and match 2D tracks.

        Errors are normalized by the image diagonal so the loss is O(1) and
        compatible with L_recon / L_attn magnitudes.

        pred_norm: (T, M, 3)
        gt_points_2d[a]: (T, M, 2)
        visible_per_view[a]: (T, M)
        """
        T, M, _ = pred_norm.shape
        W, H = image_size
        diag = float((W ** 2 + H ** 2) ** 0.5)
        total = pred_norm.new_tensor(0.0)
        count = pred_norm.new_tensor(0.0)
        for a, (K, w2c) in enumerate(zip(view_K, view_w2c_norm)):
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            # (T, M, 3)
            pts_cam = torch.einsum("ij,tmj->tmi", R, pred_norm) + t
            z = pts_cam[..., 2].clamp(min=1e-6)
            u = K[0, 0] * (pts_cam[..., 0] / z) + K[0, 2]
            v = K[1, 1] * (pts_cam[..., 1] / z) + K[1, 2]
            uv = torch.stack([u, v], dim=-1)  # (T, M, 2)
            gt = gt_points_2d[a]
            vis = visible_per_view[a].float()
            # Normalized squared pixel error (clamped: bad DLT outliers must not
            # dominate the batch and explode the running train loss).
            err = ((uv - gt) / diag).pow(2).sum(dim=-1).clamp(max=50.0)
            total = total + (err * vis).sum()
            count = count + vis.sum()
        return total / count.clamp(min=1.0)

    def attention_loss(
        self,
        attn_lists: List[List[torch.Tensor]],
        gt_norm: torch.Tensor,
        grid: torch.Tensor,
        visible: torch.Tensor,
        sigma: float = 0.1,
    ) -> torch.Tensor:
        """Encourage cross-view attention agreement via the shared grid.

        Paper: L_attn = -mean log A_{a->b}(i,i).
        We approximate A_{a->b} through the grid: for each GT point, find soft
        grid support from each view's attention, then take the product of
        attention mass at the GT's nearest grid cells across view pairs.

        attn_lists: list over T of list over views of (V, K_a) — but K_a may
        differ from M. For training we pass attention over the *reference*
        query points only when available; otherwise we build a soft target
        from GT proximity to the grid and maximize attention mass there.

        Practical implementation used here:
          For each view a and GT point m (visible), compute target distribution
          over voxels from spatial proximity of GT to grid, then CE against
          the view's attention marginalized... but attn is (V, K) over that
          view's own points.

        Simpler faithful proxy used for release:
          Build soft GT occupancy over voxels from gt_norm, then for each view
          maximize the attention-weighted occupancy (encourage attention to
          concentrate near true 3D locations). Cross-view term: encourage
          attention distributions of different views to agree on the grid.
        """
        T = len(attn_lists)
        if T == 0:
            return gt_norm.new_tensor(0.0)

        device = gt_norm.device
        # Soft GT occupancy: (T, V) from distance of each GT point to grid
        # gt_norm: (T, M, 3), grid: (V, 3)
        V = grid.shape[0]
        # Use first frame's grid (static)
        losses = []
        for t in range(T):
            vis_t = visible[t].float()  # (M,)
            if vis_t.sum() < 1:
                continue
            gt_t = gt_norm[t]  # (M, 3)
            # Target occupancy per voxel from visible GT
            dist2 = torch.cdist(grid, gt_t).pow(2)  # (V, M)
            # Soft assignment of GT to voxels
            assign = F.softmax(-dist2 / (sigma ** 2), dim=0)  # (V, M)
            target = (assign * vis_t.unsqueeze(0)).sum(dim=1)  # (V,)
            target = target / target.sum().clamp(min=1e-8)

            view_attns = attn_lists[t]
            # Per-view attention mass over voxels: sum over points
            view_masses = []
            for attn in view_attns:
                # attn: (V, K) — mass per voxel
                mass = attn.sum(dim=1)  # (V,)
                mass = mass / mass.sum().clamp(min=1e-8)
                view_masses.append(mass)
                # Align with target (stable CE)
                losses.append(
                    F.kl_div((mass + 1e-8).log(), target, reduction="sum").clamp(max=10.0)
                )

            # Cross-view agreement (soft; clamped to avoid a large loss floor)
            for i in range(len(view_masses)):
                for j in range(i + 1, len(view_masses)):
                    agree = (view_masses[i] * view_masses[j]).sum().clamp(min=1e-4)
                    # 1 - cosine-like overlap, in [0, 1]
                    losses.append(1.0 - agree)

        if not losses:
            return gt_norm.new_tensor(0.0)
        return torch.stack(losses).mean()

    def visibility_loss(
        self,
        vis_logits: torch.Tensor,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            vis_logits, visible.float()
        )

    def forward(
        self,
        pred_norm: torch.Tensor,
        gt_norm: torch.Tensor,
        visible: torch.Tensor,
        gt_points_2d: List[torch.Tensor],
        view_K: List[torch.Tensor],
        view_w2c_norm: List[torch.Tensor],
        visible_per_view: List[torch.Tensor],
        attn_lists: List[List[torch.Tensor]],
        grid: torch.Tensor,
        vis_logits: Optional[torch.Tensor] = None,
        aabb_center: Optional[torch.Tensor] = None,
        aabb_half: Optional[torch.Tensor] = None,
        view_w: Optional[torch.Tensor] = None,
        obs_points_2d: Optional[List[torch.Tensor]] = None,
        image_size: Tuple[int, int] = (640, 360),
    ) -> Dict[str, torch.Tensor]:
        l_recon = self.reconstruction_loss(
            pred_norm, gt_norm, visible, aabb_half=aabb_half, aabb_center=aabb_center
        )
        img_w, img_h = image_size
        l_proj = self.projection_loss(
            pred_norm,
            gt_points_2d,
            view_K,
            view_w2c_norm,
            visible_per_view,
            image_size=(img_w, img_h),
        )
        l_attn = self.attention_loss(attn_lists, gt_norm, grid, visible)
        total = (
            self.lambda_recon * l_recon
            + self.lambda_proj * l_proj
            + self.lambda_attn * l_attn
        )
        out = {
            "loss": total,
            "l_recon": l_recon.detach(),
            "l_proj": l_proj.detach(),
            "l_attn": l_attn.detach(),
        }
        if view_w is not None and obs_points_2d is not None:
            l_view = self.view_weight_loss(
                view_w,
                obs_points_2d,
                gt_norm,
                view_K,
                view_w2c_norm,
                visible_per_view,
                image_size=(img_w, img_h),
            )
            out["loss"] = out["loss"] + self.lambda_view * l_view
            out["l_view"] = l_view.detach()
        if vis_logits is not None:
            l_vis = self.visibility_loss(vis_logits, visible)
            out["loss"] = out["loss"] + self.lambda_vis * l_vis
            out["l_vis"] = l_vis.detach()
        return out
