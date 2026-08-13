#!/usr/bin/env python3
"""Download / selectively extract PointOdyssey robot sequences.

Keeps only rgbs + depths + anno (+ scene_info) to save disk. After each archive
is processed it can be deleted with --cleanup.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO = "aharley/pointodyssey"
KEEP_DIRS = {"rgbs", "depths"}
KEEP_FILES = {"anno.npz", "scene_info.json", "info.npz"}


def is_robot_scene(name: str) -> bool:
    # r1_new_f, r4_new_, robot_*, etc.
    if name.endswith(".mp4") or name.endswith(".py"):
        return False
    return name.startswith("r") and any(ch.isdigit() for ch in name[:3])


def extract_robots_from_tar(
    tar_path: Path,
    out_root: Path,
    split_prefix: str = "",
) -> list:
    """Extract robot sequences; return list of scene names."""
    out_root.mkdir(parents=True, exist_ok=True)
    scenes = set()
    extracted = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        for m in tqdm(members, desc=f"extract {tar_path.name}"):
            name = m.name
            # normalize: strip leading split folder (val/, test/, train/, sample/)
            parts = name.split("/")
            if not parts:
                continue
            # find scene token
            if split_prefix and parts[0] == split_prefix.rstrip("/"):
                parts = parts[1:]
            if not parts:
                continue
            scene = parts[0]
            if not is_robot_scene(scene):
                continue
            if len(parts) == 1:
                continue
            kind = parts[1]
            if kind in KEEP_FILES and len(parts) == 2:
                pass
            elif kind in KEEP_DIRS and len(parts) == 3:
                pass
            else:
                continue
            m.name = "/".join([scene] + parts[1:])
            tar.extract(m, path=out_root)
            scenes.add(scene)
            extracted += 1
    print(f"{tar_path.name}: extracted {extracted} files for scenes {sorted(scenes)}")
    return sorted(scenes)


def download_and_extract(
    splits: list,
    hf_dir: Path,
    out_root: Path,
    cleanup: bool,
):
    hf_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "sample": ("sample.tar.gz", "sample"),
        "val": ("val.tar.gz", "val"),
        "test": ("test.tar.gz", "test"),
    }
    for split in splits:
        if split == "train":
            # Multi-part archive — concatenate then extract
            parts = [
                "train.tar.gz.partaa",
                "train.tar.gz.partab",
                "train.tar.gz.partac",
                "train.tar.gz.partad",
            ]
            local_parts = []
            for part in parts:
                print(f"Downloading {part} ...")
                p = hf_hub_download(
                    REPO, part, repo_type="dataset", local_dir=str(hf_dir)
                )
                local_parts.append(Path(p))
            merged = hf_dir / "train.tar.gz"
            if not merged.exists():
                print(f"Concatenating train parts → {merged}")
                with open(merged, "wb") as out:
                    for part in local_parts:
                        with open(part, "rb") as inp:
                            while True:
                                chunk = inp.read(1024 * 1024 * 64)
                                if not chunk:
                                    break
                                out.write(chunk)
                if cleanup:
                    for part in local_parts:
                        part.unlink(missing_ok=True)
            extract_robots_from_tar(merged, out_root, split_prefix="train")
            if cleanup:
                merged.unlink(missing_ok=True)
            continue

        fname, prefix = mapping[split]
        print(f"Downloading {fname} ...")
        p = Path(
            hf_hub_download(REPO, fname, repo_type="dataset", local_dir=str(hf_dir))
        )
        extract_robots_from_tar(p, out_root, split_prefix=prefix)
        if cleanup:
            p.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["sample", "val", "test"],
        choices=["sample", "val", "test", "train"],
    )
    parser.add_argument("--hf_dir", default="./data/pointodyssey/hf")
    parser.add_argument("--out_root", default="./data/pointodyssey/raw")
    parser.add_argument("--cleanup", action="store_true", help="Delete archives after extract")
    args = parser.parse_args()
    download_and_extract(args.splits, Path(args.hf_dir), Path(args.out_root), args.cleanup)


if __name__ == "__main__":
    main()
