"""Joint TAPVid-3D-MC + PointOdyssey-MC dataset (same batch schema)."""

from __future__ import annotations

import random
from typing import Optional, Sequence

from torch.utils.data import Dataset

from lapa.data.mc_dataset import TAPVid3DMCDataset
from lapa.data.odyssey_mc_dataset import PointOdysseyMCDataset


class JointMCDataset(Dataset):
    """Alternates / randomly mixes samples from both MC datasets.

    Both underlying datasets already return the LAPA training batch dict, so
    no collation changes are needed.
    """

    def __init__(
        self,
        tapvid_mc_dir: str = "./data/tapvid3d_mc",
        tapvid_feature_dir: str = "./data/feature_cache",
        odyssey_mc_dir: str = "./data/pointodyssey_mc",
        odyssey_feature_dir: str = "./data/feature_cache_odyssey",
        data_root: str = "./data",
        split: str = "train",
        num_views: int = 3,
        num_frames: int = 24,
        max_points: int = 64,
        tapvid_prob: float = 0.5,
        use_gt_tracks: bool = True,
        scenes: Optional[Sequence[str]] = None,
    ):
        self.tapvid_prob = float(tapvid_prob)
        self.fixed_sample = None
        self.tapvid = TAPVid3DMCDataset(
            mc_dir=tapvid_mc_dir,
            feature_dir=tapvid_feature_dir,
            data_root=data_root,
            split=split,
            num_views=num_views,
            num_frames=num_frames,
            max_points=max_points,
            use_gt_tracks=use_gt_tracks,
            scenes=scenes,
        )
        self.odyssey = PointOdysseyMCDataset(
            mc_dir=odyssey_mc_dir,
            feature_dir=odyssey_feature_dir,
            split=split,
            num_views=num_views,
            num_frames=num_frames,
            max_points=max_points,
            use_gt_tracks=use_gt_tracks,
        )
        # Virtual length — same convention as single-dataset loaders
        self.length = 2000 if split == "train" else 200

    @property
    def scene_list(self):
        return list(self.tapvid.scene_list) + list(self.odyssey.scene_list)

    @property
    def scene_cams(self):
        # Used only for overfit scene checks in train_lapa
        out = dict(self.tapvid.scene_cams)
        out.update(self.odyssey.scene_cams)
        return out

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        if self.fixed_sample is not None:
            sample = self.fixed_sample
            import torch

            return {
                k: (
                    v.clone()
                    if torch.is_tensor(v)
                    else [
                        x.clone()
                        if torch.is_tensor(x)
                        else [
                            y.clone() if torch.is_tensor(y) else y
                            for y in x
                        ]
                        if isinstance(x, list)
                        else x
                        for x in v
                    ]
                    if isinstance(v, list)
                    else v
                )
                for k, v in sample.items()
            }

        rng = random.Random(index * 10007 + 17)
        use_tapvid = rng.random() < self.tapvid_prob
        src = self.tapvid if use_tapvid else self.odyssey
        # Different index space so each source sees diverse windows
        sample = src[index if use_tapvid else index + 9973]
        sample = dict(sample)
        sample["source_dataset"] = "tapvid3d" if use_tapvid else "pointodyssey"
        return sample
