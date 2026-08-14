#!/usr/bin/env bash
# Wait for PointOdyssey-MC training to finish, then publish weights (HF + GH) and update README.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs/runs_odyssey
LOG=outputs/runs_odyssey/autopublish.log
exec >>"$LOG" 2>&1
echo "=== autopublish started $(date) ==="

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

while true; do
  python3 - <<'PY'
import json
from pathlib import Path
root = Path("checkpoints/lapa_odyssey_mgpu")
rows_all = []
for run in sorted(root.glob("run_seed*")):
    h = run / "history.json"
    if not h.exists():
        print(f"{run.name}: no history")
        rows_all.append(0)
        continue
    rows = json.loads(h.read_text())
    best = min((r.get("val_l_recon", 1e9) for r in rows), default=1e9)
    last = rows[-1]
    print(f"{run.name}: ep={len(rows)}/50 best_val={best:.4f} train={last.get('train_l_recon', float('nan')):.4f}")
    rows_all.append(len(rows))
print("min_epoch", min(rows_all) if rows_all else 0)
PY
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
  echo "alive=$alive $(date +%H:%M:%S)"
  if [ "$min_ep" -ge 50 ] || { [ "$alive" = "0" ] && [ "$min_ep" -ge 48 ]; }; then
    echo "TRAINING_COMPLETE min_ep=$min_ep"
    break
  fi
  sleep 600
done

export HF_TOKEN
export GITHUB_TOKEN
HF_TOKEN="$(load_env_var HF_TOKEN)"
GITHUB_TOKEN="$(load_env_var GITHUB_TOKEN)"

echo "Selecting best checkpoint..."
/home/bi.ga/.conda/envs/lapa/bin/python scripts/select_best_and_publish.py \
  --runs_dir checkpoints/lapa_odyssey_mgpu \
  --export_dir checkpoints/lapa_odyssey_release \
  --dataset_name PointOdyssey-MC \
  --hf_repo bishoygaloaa/LAPA-PointOdyssey-MC \
  --gh_tag lapa-odyssey-weights

cat > checkpoints/lapa_odyssey_release/README.md <<'EOF'
---
license: mit
library_name: pytorch
tags:
  - point-tracking
  - multi-camera
  - lapa
---

# LAPA

Pretrained weights for [LAPA](https://arxiv.org/abs/2512.04213) (Look Around and Pay Attention) on PointOdyssey-MC.

## Load

```python
import torch
from lapa.models.lapa import LAPA

ckpt = torch.load("lapa.pt", map_location="cpu")
model = LAPA()
model.load_state_dict(ckpt["model"])
model.eval()
```

Code: https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-
EOF

echo "Uploading to Hugging Face..."
/home/bi.ga/.conda/envs/lapa/bin/python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo = "bishoygaloaa/LAPA-PointOdyssey-MC"
export = Path("checkpoints/lapa_odyssey_release")
api.create_repo(repo, exist_ok=True, private=False, repo_type="model")
api.upload_folder(
    folder_path=str(export),
    repo_id=repo,
    repo_type="model",
    commit_message="Add LAPA pretrained weights (PointOdyssey-MC)",
)
print("Published https://huggingface.co/" + repo)
print("files:", api.list_repo_files(repo, repo_type="model"))
PY

echo "Creating GitHub release..."
DEST=checkpoints/lapa_odyssey_release/lapa.pt
REPO=ostadabbas/Look-Around-and-Pay-Attention-LAPA-
TAG=lapa-odyssey-weights

REL_ID=$(curl -sS -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/releases/tags/$TAG" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id') or '')")
if [ -n "$REL_ID" ]; then
  curl -sS -X DELETE -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/releases/$REL_ID" >/dev/null
fi

CREATE=$(curl -sS -X POST -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/releases" \
  -d "{\"tag_name\":\"$TAG\",\"name\":\"LAPA pretrained weights (PointOdyssey-MC)\",\"body\":\"Pretrained LAPA weights for PointOdyssey-MC.\\n\\nPaper: https://arxiv.org/abs/2512.04213\",\"draft\":false,\"prerelease\":false}")
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
import re
from pathlib import Path

p = Path("README.md")
text = p.read_text()
# Mark PointOdyssey weights as live in the table
text = text.replace(
    "| LAPA PointOdyssey-MC | [bishoygaloaa/LAPA-PointOdyssey-MC](https://huggingface.co/bishoygaloaa/LAPA-PointOdyssey-MC) *(weights uploading when training finishes)* | Coming soon |",
    "| LAPA PointOdyssey-MC | [bishoygaloaa/LAPA-PointOdyssey-MC](https://huggingface.co/bishoygaloaa/LAPA-PointOdyssey-MC) | [lapa-odyssey-weights](https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-/releases/tag/lapa-odyssey-weights) (`lapa.pt`) |",
)

if "LAPA-PointOdyssey-MC/resolve/main/lapa.pt" not in text:
    insert = (
        "**Direct download (TAPVid-3D-MC):**\n"
        "- Hugging Face: https://huggingface.co/bishoygaloaa/LAPA-TAPVid-3D-MC/resolve/main/lapa.pt\n"
        "- GitHub: https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-/releases/download/lapa-weights/lapa.pt\n\n"
        "**Direct download (PointOdyssey-MC):**\n"
        "- Hugging Face: https://huggingface.co/bishoygaloaa/LAPA-PointOdyssey-MC/resolve/main/lapa.pt\n"
        "- GitHub: https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-/releases/download/lapa-odyssey-weights/lapa.pt"
    )
    text2 = re.sub(
        r"\*\*Direct download \(TAPVid-3D-MC\):\*\*\n- Hugging Face:.*?\n- GitHub:.*?(?=\n\n)",
        insert,
        text,
        count=1,
        flags=re.S,
    )
    text = text2 if text2 != text else text

p.write_text(text)
print("README updated")
PY

git add README.md
git commit -m "$(cat <<'EOF'
Add PointOdyssey-MC pretrained weight links to README.

EOF
)" || echo "commit skipped (maybe no changes)"

git push "https://x-access-token:${GITHUB_TOKEN}@github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-.git" HEAD:master
echo "=== AUTOPUBLISH_DONE $(date) ==="
