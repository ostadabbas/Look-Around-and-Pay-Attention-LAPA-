#!/usr/bin/env python3
"""Download TAPVid-3D pstudio (minival + full_eval) and Dynamic3DGaussians data.zip."""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

SCENES = ("basketball", "boxes", "football", "juggle", "softball", "tennis")

# Official TAPVid-3D release roots
GCS_MINIVAL = "https://storage.googleapis.com/dm-tapnet/tapvid3d/release_files/minival_v1.0/"
GCS_FULL_EVAL = "https://storage.googleapis.com/dm-tapnet/tapvid3d/release_files/full_eval_v1.0/"

# Dynamic3DGaussians Panoptic Studio release (frames + train_meta.json with k/w2c)
D3G_URL = "https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip"

SPLITS_URL = (
    "https://raw.githubusercontent.com/google-deepmind/tapnet/main/"
    "tapnet/tapvid3d/splits/tapvid3d_splits.py"
)


def fetch_splits_text() -> str:
    resp = requests.get(SPLITS_URL, timeout=60)
    resp.raise_for_status()
    return resp.text


def parse_pstudio_files(splits_text: str, split_name: str) -> list[str]:
    """Extract pstudio .npz filenames from the official splits file."""
    marker = f"{split_name} = {{"
    start = splits_text.index(marker)
    # Find matching closing brace at same indent level (dict ends at \n})
    depth = 0
    i = start + len(marker) - 1
    while i < len(splits_text):
        ch = splits_text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                body = splits_text[start : i + 1]
                break
        i += 1
    else:
        raise RuntimeError(f"Could not parse {split_name}")

    all_files = re.findall(r'"([^"]+\.npz)"', body)
    pstudio = [f for f in all_files if f.split("_")[0] in SCENES]
    return sorted(set(pstudio))


def download_file(url: str, dest: Path, skip_existing: bool = True) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        return True
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            if r.status_code != 200:
                print(f"FAIL {r.status_code}: {url}")
                return False
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=dest.name,
                leave=False,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        tmp.replace(dest)
        return True
    except Exception as e:
        print(f"ERROR downloading {url}: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def download_npz_list(files: list[str], base_url: str, out_dir: Path, workers: int = 8) -> None:
    tasks = []
    for fname in files:
        scene = fname.split("_")[0]
        dest = out_dir / f"tap3d_{scene}" / fname
        url = base_url + fname
        tasks.append((url, dest))

    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_file, u, d): (u, d) for u, d in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"npz@{base_url.split('/')[-2]}"):
            if fut.result():
                ok += 1
            else:
                fail += 1
    print(f"Done: {ok} ok, {fail} failed (from {base_url})")


def download_d3g(out_dir: Path) -> Path:
    zip_path = out_dir / "dynamic3dgaussians_data.zip"
    extract_dir = out_dir / "d3g"
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        print(f"Downloading Dynamic3DGaussians ({D3G_URL}) ...")
        # Use urllib for large file with progress via content-length
        ok = download_file(D3G_URL, zip_path, skip_existing=False)
        if not ok:
            raise RuntimeError("Failed to download Dynamic3DGaussians data.zip")
    else:
        print(f"Already have {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    if not (extract_dir / "data").exists():
        print(f"Extracting to {extract_dir} ...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    else:
        print(f"Already extracted at {extract_dir / 'data'}")
    return extract_dir / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip_d3g", action="store_true")
    parser.add_argument("--skip_npz", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_npz:
        print("Fetching official TAPVid-3D splits ...")
        splits_text = fetch_splits_text()
        minival = parse_pstudio_files(splits_text, "MINIVAL_FILES")
        full_eval = parse_pstudio_files(splits_text, "FULL_EVAL_FILES")
        print(f"minival pstudio: {len(minival)} files")
        print(f"full_eval pstudio: {len(full_eval)} files")

        # Save file lists for later
        (out / "minival_pstudio.txt").write_text("\n".join(minival) + "\n")
        (out / "full_eval_pstudio.txt").write_text("\n".join(full_eval) + "\n")

        download_npz_list(minival, GCS_MINIVAL, out / "tapvid3d_minival", workers=args.workers)
        download_npz_list(full_eval, GCS_FULL_EVAL, out / "tapvid3d_full_eval", workers=args.workers)

    if not args.skip_d3g:
        d3g_data = download_d3g(out)
        print(f"Dynamic3DGaussians data root: {d3g_data}")
        # Quick listing
        scenes = sorted([p.name for p in d3g_data.iterdir() if p.is_dir()])
        print(f"Scenes in D3G: {scenes}")

    print("All downloads complete.")


if __name__ == "__main__":
    main()
