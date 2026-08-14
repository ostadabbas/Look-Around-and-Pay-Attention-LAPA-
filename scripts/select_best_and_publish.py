#!/usr/bin/env python3
"""Pick the best checkpoint among parallel runs and publish the weights.

Usage (TAPVid-3D-MC):
  python scripts/select_best_and_publish.py \
      --runs_dir checkpoints/lapa_mgpu \
      --export_dir checkpoints/lapa_release \
      --publish_hf --hf_repo bishoygaloaa/lapa-tapvid3d-mc \
      --publish_gh --gh_tag lapa-weights

Usage (PointOdyssey-MC):
  python scripts/select_best_and_publish.py \
      --runs_dir checkpoints/lapa_odyssey_mgpu \
      --export_dir checkpoints/lapa_odyssey_release \
      --hf_repo bishoygaloaa/lapa-pointodyssey-mc \
      --gh_tag lapa-odyssey-weights \
      --dataset_name PointOdyssey-MC \
      --publish_hf --publish_gh
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch


def find_best_run(runs_dir: Path) -> Path:
    best_path, best_score = None, None  # higher APD wins; else lower recon
    for run in sorted(runs_dir.glob("run_seed*")):
        ckpt_path = run / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = run / "last.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        apd = ckpt.get("best_apd", None)
        recon = float(ckpt.get("best_recon", 1e9))
        hist = run / "history.json"
        if hist.exists():
            rows = json.loads(hist.read_text())
            apds = [r.get("val_apd") for r in rows if r.get("val_apd") is not None]
            recons = [r.get("val_l_recon") for r in rows if r.get("val_l_recon") is not None]
            if apds:
                apd = max(apds)
            if recons:
                recon = min(recons)
        print(f"  {run.name}: apd={apd} recon={recon:.4f} epoch={ckpt.get('epoch')}")
        if apd is not None:
            score = (1, float(apd), -recon)  # prefer APD
        else:
            score = (0, -recon, 0.0)
        if best_score is None or score > best_score:
            best_score = score
            best_path = ckpt_path
    if best_path is None:
        raise RuntimeError(f"No checkpoints found under {runs_dir}")
    print(f"Selected {best_path} (score={best_score})")
    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="checkpoints/lapa_mgpu")
    parser.add_argument("--export_dir", default="checkpoints/lapa_release")
    parser.add_argument("--dataset_name", default="TAPVid-3D-MC")
    parser.add_argument("--publish_hf", action="store_true")
    parser.add_argument("--hf_repo", default="bishoygaloaa/lapa-tapvid3d-mc")
    parser.add_argument("--publish_gh", action="store_true")
    parser.add_argument("--gh_tag", default="lapa-weights")
    args = parser.parse_args()

    best_ckpt = find_best_run(Path(args.runs_dir))
    export = Path(args.export_dir)
    export.mkdir(parents=True, exist_ok=True)

    dest = export / "lapa.pt"
    shutil.copy2(best_ckpt, dest)
    shutil.copy2(best_ckpt, export / "best.pt")

    readme = export / "README.md"
    readme.write_text(
        "# LAPA\n\n"
        "Pretrained weights for "
        "[LAPA](https://arxiv.org/abs/2512.04213) "
        f"(Look Around and Pay Attention) on {args.dataset_name}.\n\n"
        "## Load\n\n"
        "```python\n"
        "import torch\n"
        "from lapa.models.lapa import LAPA\n\n"
        "ckpt = torch.load('lapa.pt', map_location='cpu')\n"
        "model = LAPA()\n"
        "model.load_state_dict(ckpt['model'])\n"
        "model.eval()\n"
        "```\n\n"
        "Code: https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-\n"
    )
    print(f"Exported weights to {dest}")

    if args.publish_hf:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.hf_repo, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=str(export),
            repo_id=args.hf_repo,
            repo_type="model",
            commit_message=f"Add LAPA pretrained weights ({args.dataset_name})",
        )
        print(f"Published to https://huggingface.co/{args.hf_repo}")

    if args.publish_gh:
        tag = args.gh_tag
        subprocess.call(["gh", "release", "delete", tag, "--yes"], stderr=subprocess.DEVNULL)
        subprocess.check_call(
            [
                "gh",
                "release",
                "create",
                tag,
                str(dest),
                "--title",
                f"LAPA pretrained weights ({args.dataset_name})",
                "--notes",
                f"Pretrained LAPA weights for {args.dataset_name}.\n\n"
                "Paper: https://arxiv.org/abs/2512.04213",
            ]
        )
        print(f"GitHub release '{tag}' created.")


if __name__ == "__main__":
    main()
