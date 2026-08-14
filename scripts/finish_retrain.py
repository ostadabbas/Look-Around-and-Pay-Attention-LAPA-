#!/usr/bin/env python3
"""After retraining: pick best checkpoints, eval TAPVid minival, publish, patch README."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_best_and_publish import find_best_run

ROOT = Path(__file__).resolve().parents[1]


def load_env():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if not line.strip() or line.strip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def export_and_publish(runs_dir, export_dir, dataset_name, hf_repo, gh_tag):
    from huggingface_hub import HfApi
    import shutil

    best = find_best_run(Path(runs_dir))
    export = Path(export_dir)
    export.mkdir(parents=True, exist_ok=True)
    dest = export / "lapa.pt"
    shutil.copy2(best, dest)
    shutil.copy2(best, export / "best.pt")
    meta = {
        "dataset": dataset_name,
        "source_checkpoint": str(best),
        "protocol": "query-frame GT 2D; t>0 CoTracker when cached",
    }
    try:
        ckpt = __import__("torch").load(best, map_location="cpu")
        meta["best_apd"] = float(ckpt.get("best_apd", -1))
        meta["best_recon"] = float(ckpt.get("best_recon", -1))
        meta["epoch"] = int(ckpt.get("epoch", -1))
    except Exception as e:
        meta["ckpt_meta_error"] = str(e)
    (export / "release_meta.json").write_text(json.dumps(meta, indent=2))
    (export / "README.md").write_text(
        "# LAPA\n\n"
        f"Pretrained weights for [LAPA](https://arxiv.org/abs/2512.04213) "
        f"on {dataset_name}.\n\n"
        "Query-frame 2D is GT; later frames use CoTracker tracks.\n"
        "Code: https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-\n"
        f"\nRelease meta: `{meta}`\n"
    )
    api = HfApi()
    api.create_repo(hf_repo, exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(export),
        repo_id=hf_repo,
        repo_type="model",
        commit_message=f"Update LAPA weights ({dataset_name})",
    )
    print(f"HF: https://huggingface.co/{hf_repo}")
    # Replace GitHub release (delete may fail if missing)
    subprocess.call(
        ["gh", "release", "delete", gh_tag, "--yes"],
        stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        [
            "gh",
            "release",
            "create",
            gh_tag,
            str(dest),
            "--title",
            f"LAPA pretrained weights ({dataset_name})",
            "--notes",
            f"Pretrained LAPA weights for {dataset_name}.\n"
            f"best_apd={meta.get('best_apd')}\n"
            "Paper: https://arxiv.org/abs/2512.04213",
        ]
    )
    print(f"GitHub release: {gh_tag}")
    return dest


def eval_tapvid(ckpt: Path, use_gt: bool, out_json: Path) -> dict:
    cmd = [
        "python",
        "evaluate_lapa.py",
        "--checkpoint",
        str(ckpt),
        "--output",
        str(out_json),
        "--device",
        "cuda:1",
    ]
    if use_gt:
        cmd.append("--use_gt_tracks")
    subprocess.check_call(cmd, cwd=str(ROOT))
    return json.loads(out_json.read_text())


def patch_readme(metrics: dict, dlt_gt: dict, dlt_ct: dict | None) -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    block = (
        "\n### Measured minival numbers (full-sequence, n=50)\n\n"
        "| Method | APD | OA | 3D-AJ | 2D-AJ |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| DLT (GT 2D) | {dlt_gt['APD']:.1f} | {dlt_gt['OA']:.1f} | {dlt_gt['AJ3D']:.1f} | {dlt_gt['AJ2D']:.1f} |\n"
    )
    if dlt_ct is not None:
        block += (
            f"| DLT (CoTracker 2D) | {dlt_ct['APD']:.1f} | {dlt_ct['OA']:.1f} | "
            f"{dlt_ct['AJ3D']:.1f} | {dlt_ct['AJ2D']:.1f} |\n"
        )
    block += (
        f"| LAPA (CoTracker 2D) | {metrics['APD']:.1f} | {metrics['OA']:.1f} | "
        f"{metrics['AJ3D']:.1f} | {metrics['AJ2D']:.1f} |\n"
        f"\nOA constant-visible baseline: {metrics.get('OA_const_vis', float('nan')):.1f}. "
        "Query frame 2D is always GT. Checkpoint: `checkpoints/lapa_release/lapa.pt` "
        "(same file as the Hugging Face / GitHub TAPVid-3D-MC release).\n"
    )
    marker = "### Evaluate (TAPVid-3D-MC minival, Table 2 protocol)"
    if "### Measured minival numbers" in text:
        # Replace the previous measured block through the Evaluate heading.
        import re

        text = re.sub(
            r"\n### Measured minival numbers \(full-sequence, n=50\).*?(?=\n### Evaluate \(TAPVid-3D-MC minival)",
            block + "\n",
            text,
            flags=re.S,
        )
    else:
        text = text.replace(marker, block + "\n" + marker)
    readme.write_text(text)


def main():
    load_env()
    os.chdir(ROOT)
    os.environ.setdefault("PYTHONPATH", str(ROOT))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if os.environ.get("GITHUB_TOKEN"):
        os.environ["GH_TOKEN"] = os.environ["GITHUB_TOKEN"]
    if os.environ.get("HF_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    tapvid = export_and_publish(
        "checkpoints/lapa_mgpu",
        "checkpoints/lapa_release",
        "TAPVid-3D-MC",
        "bishoygaloaa/LAPA-TAPVid-3D-MC",
        "lapa-weights",
    )
    export_and_publish(
        "checkpoints/lapa_odyssey_mgpu",
        "checkpoints/lapa_odyssey_release",
        "PointOdyssey-MC",
        "bishoygaloaa/LAPA-PointOdyssey-MC",
        "lapa-odyssey-weights",
    )
    export_and_publish(
        "checkpoints/lapa_joint_mgpu",
        "checkpoints/lapa_joint_release",
        "Joint TAPVid-3D-MC + PointOdyssey-MC",
        "bishoygaloaa/LAPA-Joint",
        "lapa-joint-weights",
    )

    metrics = eval_tapvid(tapvid, use_gt=False, out_json=Path("outputs/eval_metrics.json"))
    dlt_gt = json.loads(Path("outputs/dlt_baseline_gt.json").read_text())
    dlt_ct_path = Path("outputs/dlt_baseline_cotracker.json")
    subprocess.call(
        [
            "python",
            "scripts/triangulation_baseline.py",
            "--output",
            str(dlt_ct_path),
        ],
        cwd=str(ROOT),
    )
    dlt_ct = json.loads(dlt_ct_path.read_text()) if dlt_ct_path.exists() else None
    patch_readme(metrics, dlt_gt, dlt_ct)

    reply = Path("outputs/reporter_reply.txt")
    reply.write_text(
        "Hello Frano,\n\n"
        "Thank you for the careful eval — your numbers were correct for the first "
        "public checkpoint. The evaluation harness and the model inputs have been "
        "fixed (corresponded multi-view tracks + geometric DLT anchor).\n\n"
        "To reproduce Table 2 on TAPVid-3D-MC minival:\n\n"
        "```bash\n"
        "hf download bishoygaloaa/LAPA-TAPVid-3D-MC lapa.pt --local-dir checkpoints/lapa\n"
        "python evaluate_lapa.py --checkpoint checkpoints/lapa/lapa.pt \\\n"
        "  --output outputs/eval_metrics.json\n"
        "```\n\n"
        "Protocol:\n"
        "1. Exact command: the evaluate_lapa.py invocation above (full-sequence minival, "
        "3 views, query frame = first frame).\n"
        "2. Use CoTracker tracks at t>0 (the default). Do not pass --use_gt_tracks; that "
        "flag is an upper bound that feeds GT 2D projections. Only the query frame is GT.\n"
        f"3. Checkpoint: the updated lapa.pt on Hugging Face "
        f"(bishoygaloaa/LAPA-TAPVid-3D-MC) / GitHub release lapa-weights. "
        f"Measured minival: APD {metrics['APD']:.1f}, OA {metrics['OA']:.1f}, "
        f"3D-AJ {metrics['AJ3D']:.1f}, 2D-AJ {metrics['AJ2D']:.1f}.\n\n"
        "Best regards,\n"
        "Bishoy\n"
    )
    print(f"Wrote {reply}")
    subprocess.call(["git", "add", "README.md"], cwd=str(ROOT))
    subprocess.call(
        [
            "git",
            "commit",
            "-m",
            "Update README with TAPVid-3D-MC minival eval protocol and measured numbers.",
        ],
        cwd=str(ROOT),
    )
    Path("outputs/publish_done.flag").write_text(
        json.dumps(
            {
                "APD": metrics["APD"],
                "OA": metrics["OA"],
                "AJ3D": metrics["AJ3D"],
                "AJ2D": metrics["AJ2D"],
            },
            indent=2,
        )
        + "\n"
    )
    print("finish_retrain done")


if __name__ == "__main__":
    main()
