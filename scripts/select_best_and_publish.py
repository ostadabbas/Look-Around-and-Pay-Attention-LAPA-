#!/usr/bin/env python3
"""Export a trained checkpoint and optionally publish weights.

Usage:
  python scripts/select_best_and_publish.py \
      --checkpoint checkpoints/lapa/best.pt \
      --export_dir checkpoints/lapa_release \
      --publish_hf --hf_repo bishoygaloaa/LAPA-TAPVid-3D-MC \
      --publish_gh --gh_tag lapa-weights
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt / lapa.pt")
    parser.add_argument("--export_dir", default="checkpoints/lapa_release")
    parser.add_argument("--dataset_name", default="TAPVid-3D-MC")
    parser.add_argument("--publish_hf", action="store_true")
    parser.add_argument("--hf_repo", default="bishoygaloaa/LAPA-TAPVid-3D-MC")
    parser.add_argument("--publish_gh", action="store_true")
    parser.add_argument("--gh_tag", default="lapa-weights")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    export = Path(args.export_dir)
    export.mkdir(parents=True, exist_ok=True)
    dest = export / "lapa.pt"
    shutil.copy2(ckpt, dest)
    shutil.copy2(ckpt, export / "best.pt")

    (export / "README.md").write_text(
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
