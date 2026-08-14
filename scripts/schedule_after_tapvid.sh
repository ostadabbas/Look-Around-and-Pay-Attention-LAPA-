#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lapa
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$ROOT"

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
    # Also treat live train_lapa as alive (pid files can go stale)
    if pgrep -f 'python train_lapa.py' >/dev/null 2>&1; then
      # only if this wave's output dirs are being written recently, else fall through
      :
    fi
    if [ "$alive" -eq 0 ]; then
      break
    fi
    sleep 60
  done
}

wait_pids outputs/runs
echo "[schedule] TAPVid training finished"

# Clear stale Odyssey/Joint runs so we never publish pre-fix APD=0 weights
rm -rf checkpoints/lapa_odyssey_mgpu/run_seed* checkpoints/lapa_joint_mgpu/run_seed*
bash scripts/launch_4gpu_train_odyssey.sh
wait_pids outputs/runs_odyssey
echo "[schedule] Odyssey training finished"

bash scripts/launch_4gpu_train_joint.sh
wait_pids outputs/runs_joint
echo "[schedule] Joint training finished"

echo "[schedule] publishing to HF + GitHub ..."
python scripts/finish_retrain.py
date > outputs/publish_done.flag
echo "[schedule] publish/eval complete"
