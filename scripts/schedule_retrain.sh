#!/usr/bin/env bash
# Wait for CoTracker caches, then train TAPVid → Odyssey → Joint (4 GPUs each).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lapa

need_canonical=156
need_eval=140
echo "[schedule] waiting for CoTracker caches ..."
while true; do
  nc=$(find data/feature_cache_canonical -name 'cam_*.h5' 2>/dev/null | wc -l)
  ne=$(find data/feature_cache_eval -name 'ref*_cam*.h5' 2>/dev/null | wc -l)
  echo "[schedule] canonical $nc / $need_canonical   eval $ne / $need_eval"
  if [ "$nc" -ge "$need_canonical" ] && [ "$ne" -ge "$need_eval" ]; then
    break
  fi
  sleep 30
done

echo "[schedule] launching TAPVid-3D-MC 4-GPU train"
bash scripts/launch_4gpu_train.sh

wait_pids() {
  local dir="$1"
  echo "[schedule] polling PIDs in $dir"
  while true; do
    alive=0
    for f in "$dir"/seed*.pid; do
      [ -f "$f" ] || continue
      pid=$(cat "$f")
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        alive=1
      fi
    done
    if [ "$alive" -eq 0 ]; then
      break
    fi
    sleep 60
  done
}

wait_pids outputs/runs
echo "[schedule] TAPVid training finished"

echo "[schedule] launching PointOdyssey-MC 4-GPU train"
bash scripts/launch_4gpu_train_odyssey.sh
wait_pids outputs/runs_odyssey
echo "[schedule] Odyssey training finished"

echo "[schedule] launching Joint 4-GPU train"
bash scripts/launch_4gpu_train_joint.sh
wait_pids outputs/runs_joint
echo "[schedule] Joint training finished"

echo "[schedule] selecting checkpoints + TAPVid minival eval + publish"
python scripts/finish_retrain.py
echo "[schedule] all training waves complete"
