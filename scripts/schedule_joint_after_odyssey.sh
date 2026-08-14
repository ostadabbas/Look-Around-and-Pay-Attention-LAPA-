#!/usr/bin/env bash
# After Odyssey training finishes + publishes, train the joint model and publish it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs/runs_joint
LOG=outputs/runs_joint/schedule.log
exec >>"$LOG" 2>&1
echo "=== joint schedule started $(date) ==="

load_env_var() {
  local key="$1"
  python3 - "$key" <<'PY'
from pathlib import Path
import sys
key = sys.argv[1] + "="
for line in Path(".env").read_text().splitlines():
    if line.startswith(key):
        print(line.split("=", 1)[1].strip().strip('"').strip("'"))
        break
PY
}

echo "Waiting for Odyssey training to finish..."
while true; do
  min_ep=$(python3 - <<'PY'
import json
from pathlib import Path
root = Path("checkpoints/lapa_odyssey_mgpu")
mins = [len(json.loads((r / "history.json").read_text())) for r in root.glob("run_seed*") if (r / "history.json").exists()]
print(min(mins) if mins else 0)
PY
)
  alive=$(pgrep -c -f "python train_lapa.py --dataset pointodyssey" 2>/dev/null || echo 0)
  alive=$(echo "$alive" | head -1)
  echo "odyssey min_ep=$min_ep alive=$alive $(date +%H:%M:%S)"
  if [ "$min_ep" -ge 50 ]; then
    echo "Odyssey training complete"
    break
  fi
  if [ "$alive" = "0" ] && [ "$min_ep" -ge 48 ]; then
    echo "Odyssey jobs stopped near completion (min_ep=$min_ep)"
    break
  fi
  sleep 600
done

# Let Odyssey autopublish finish if it's still running (best-effort wait)
for _ in $(seq 1 60); do
  if pgrep -f "scripts/autopublish_odyssey.sh" >/dev/null 2>&1; then
    echo "waiting for Odyssey autopublish... $(date +%H:%M:%S)"
    sleep 60
  else
    break
  fi
done

# Ensure GPUs are free of prior train jobs
for _ in $(seq 1 30); do
  n=$(pgrep -c -f "python train_lapa.py" 2>/dev/null || echo 0)
  n=$(echo "$n" | head -1)
  if [ "$n" = "0" ]; then
    break
  fi
  echo "waiting for GPUs free (train_lapa alive=$n)..."
  sleep 60
done

echo "Launching joint 4-GPU training..."
bash scripts/launch_4gpu_train_joint.sh

echo "Waiting for joint training to finish..."
while true; do
  min_ep=$(python3 - <<'PY'
import json
from pathlib import Path
root = Path("checkpoints/lapa_joint_mgpu")
mins = [len(json.loads((r / "history.json").read_text())) for r in root.glob("run_seed*") if (r / "history.json").exists()]
print(min(mins) if mins else 0)
PY
)
  alive=$(pgrep -c -f "python train_lapa.py --dataset joint" 2>/dev/null || echo 0)
  alive=$(echo "$alive" | head -1)
  echo "joint min_ep=$min_ep alive=$alive $(date +%H:%M:%S)"
  if [ "$min_ep" -ge 50 ]; then
    echo "Joint training complete"
    break
  fi
  if [ "$alive" = "0" ] && [ "$min_ep" -ge 48 ]; then
    echo "Joint jobs stopped near completion (min_ep=$min_ep)"
    break
  fi
  sleep 600
done

export HF_TOKEN GITHUB_TOKEN
HF_TOKEN="$(load_env_var HF_TOKEN)"
GITHUB_TOKEN="$(load_env_var GITHUB_TOKEN)"

echo "Selecting best joint checkpoint..."
/home/bi.ga/.conda/envs/lapa/bin/python scripts/select_best_and_publish.py \
  --runs_dir checkpoints/lapa_joint_mgpu \
  --export_dir checkpoints/lapa_joint_release \
  --dataset_name "TAPVid-3D-MC + PointOdyssey-MC (joint)" \
  --hf_repo bishoygaloaa/LAPA-Joint \
  --gh_tag lapa-joint-weights

cat > checkpoints/lapa_joint_release/README.md <<'EOF'
---
license: mit
library_name: pytorch
tags:
  - point-tracking
  - multi-camera
  - lapa
---

# LAPA — Joint (TAPVid-3D-MC + PointOdyssey-MC)

Pretrained [LAPA](https://arxiv.org/abs/2512.04213) weights trained jointly on TAPVid-3D-MC and PointOdyssey-MC.

Part of the [LAPA collection](https://huggingface.co/collections/bishoygaloaa/lapa-6a79dcf4bbedf556ad7da964).

## Load

```python
import torch
from huggingface_hub import hf_hub_download
from lapa.models.lapa import LAPA

path = hf_hub_download("bishoygaloaa/LAPA-Joint", "lapa.pt")
ckpt = torch.load(path, map_location="cpu")
model = LAPA()
model.load_state_dict(ckpt["model"])
model.eval()
```

Code: https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-
EOF

echo "Uploading joint weights to Hugging Face..."
/home/bi.ga/.conda/envs/lapa/bin/python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
ns = "bishoygaloaa"
repo = f"{ns}/LAPA-Joint"
export = Path("checkpoints/lapa_joint_release")
api.create_repo(repo, exist_ok=True, private=False, repo_type="model")
api.upload_folder(
    folder_path=str(export),
    repo_id=repo,
    repo_type="model",
    commit_message="Add LAPA joint pretrained weights",
)
print("Published https://huggingface.co/" + repo)

# Ensure it's in the LAPA collection
cols = list(api.list_collections(owner=ns))
slug = None
for c in cols:
    if c.title == "LAPA" or c.slug.startswith(f"{ns}/lapa"):
        slug = c.slug
        break
if slug:
    try:
        api.add_collection_item(
            slug,
            item_id=repo,
            item_type="model",
            note="LAPA trained jointly on TAPVid-3D-MC + PointOdyssey-MC",
            exists_ok=True,
        )
        print("Added to collection", slug)
    except Exception as e:
        print("collection add:", e)
PY

echo "Creating GitHub release..."
DEST=checkpoints/lapa_joint_release/lapa.pt
REPO=ostadabbas/Look-Around-and-Pay-Attention-LAPA-
TAG=lapa-joint-weights
REL_ID=$(curl -sS -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/releases/tags/$TAG" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id') or '')")
if [ -n "$REL_ID" ]; then
  curl -sS -X DELETE -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/releases/$REL_ID" >/dev/null
fi
CREATE=$(curl -sS -X POST -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/releases" \
  -d "{\"tag_name\":\"$TAG\",\"name\":\"LAPA pretrained weights (Joint)\",\"body\":\"Pretrained LAPA weights trained jointly on TAPVid-3D-MC and PointOdyssey-MC.\\n\\nPaper: https://arxiv.org/abs/2512.04213\",\"draft\":false,\"prerelease\":false}")
echo "$CREATE" | python3 -c "import sys,json; d=json.load(sys.stdin); print('release_url:', d.get('html_url')); print('err:', d.get('message'))"
UPLOAD_URL=$(echo "$CREATE" | python3 -c "import sys,json; d=json.load(sys.stdin); u=d.get('upload_url',''); print(u.split('{')[0] if u else '')")
curl -sS -X POST \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @"$DEST" \
  "${UPLOAD_URL}?name=lapa.pt" | python3 -c "import sys,json; d=json.load(sys.stdin); print('asset:', d.get('browser_download_url') or d.get('message'))"

echo "Updating README..."
python3 - <<'PY'
from pathlib import Path
p = Path("README.md")
text = p.read_text()
row = "| LAPA Joint | [bishoygaloaa/LAPA-Joint](https://huggingface.co/bishoygaloaa/LAPA-Joint) | [lapa-joint-weights](https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-/releases/tag/lapa-joint-weights) (`lapa.pt`) |"
pending = "| LAPA Joint | [bishoygaloaa/LAPA-Joint](https://huggingface.co/bishoygaloaa/LAPA-Joint) *(scheduled)* | Coming soon |"
if pending in text:
    text = text.replace(pending, row)
elif row not in text and "| LAPA PointOdyssey-MC |" in text:
    # insert after PointOdyssey row
    lines = text.splitlines()
    out = []
    for line in lines:
        out.append(line)
        if line.startswith("| LAPA PointOdyssey-MC |") and row not in "\n".join(out):
            out.append(row)
    text = "\n".join(out) + ("\n" if text.endswith("\n") else "")
if "LAPA-Joint/resolve/main/lapa.pt" not in text:
    block = (
        "\n**Direct download (Joint):**\n"
        "- Hugging Face: https://huggingface.co/bishoygaloaa/LAPA-Joint/resolve/main/lapa.pt\n"
        "- GitHub: https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-/releases/download/lapa-joint-weights/lapa.pt\n"
    )
    if "**Direct download (PointOdyssey-MC):**" in text:
        # append after PO block end — simple append before Quick Start
        text = text.replace("\n## Quick Start", block + "\n## Quick Start")
    else:
        text = text.replace("\n## Quick Start", block + "\n## Quick Start")
p.write_text(text)
print("README updated")
PY

git add README.md
git commit -m "$(cat <<'EOF'
Add LAPA joint pretrained weight links to README.

EOF
)" || echo "commit skipped"

git push "https://x-access-token:${GITHUB_TOKEN}@github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-.git" HEAD:master
echo "=== JOINT_SCHEDULE_DONE $(date) ==="
