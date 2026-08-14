#!/usr/bin/env bash
# Backup watcher: when TAPVid+Odyssey+Joint waves are done, publish if needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lapa
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$ROOT"
mkdir -p outputs

MARKER=outputs/publish_done.flag
LOG=outputs/publish_watchdog.log

log() { echo "[$(date)] $*" | tee -a "$LOG"; }

wave_done() {
  local dir="$1"
  # No live train_lapa, and every seed has a best.pt
  if pgrep -f 'python train_lapa.py' >/dev/null 2>&1; then
    return 1
  fi
  local n=0
  for s in 42 123 7 99; do
    if [ -f "$dir/run_seed${s}/best.pt" ]; then
      n=$((n + 1))
    fi
  done
  [ "$n" -eq 4 ]
}

log "watchdog started (pid $$)"

# Wait until no train_lapa and all three waves have 4 best.pt files
while true; do
  if [ -f "$MARKER" ]; then
    log "publish already done ($MARKER); exiting"
    exit 0
  fi

  tap=0; ody=0; jnt=0
  for s in 42 123 7 99; do
    [ -f "checkpoints/lapa_mgpu/run_seed${s}/best.pt" ] && tap=$((tap+1))
    [ -f "checkpoints/lapa_odyssey_mgpu/run_seed${s}/best.pt" ] && ody=$((ody+1))
    [ -f "checkpoints/lapa_joint_mgpu/run_seed${s}/best.pt" ] && jnt=$((jnt+1))
  done
  alive=$(pgrep -c -f 'python train_lapa.py' || true)
  log "alive_train=$alive tap_best=$tap/4 odyssey_best=$ody/4 joint_best=$jnt/4"

  # Only publish after Joint wave exists with fresh checkpoints AND no trainers.
  # Require Joint history to mention val_apd from the new recipe (best_apd > 0)
  # OR simply: no trainers, all 3 dirs have 4 best.pt, and schedule is gone.
  if [ "${alive:-0}" -eq 0 ] && [ "$tap" -eq 4 ] && [ "$ody" -eq 4 ] && [ "$jnt" -eq 4 ]; then
    # Require Odyssey/Joint checkpoints to be newer than TAPVid (avoid stale APD=0 publishes)
    tap_t=$(stat -c %Y checkpoints/lapa_mgpu/run_seed42/best.pt 2>/dev/null || echo 0)
    ody_t=$(stat -c %Y checkpoints/lapa_odyssey_mgpu/run_seed42/best.pt 2>/dev/null || echo 0)
    jnt_t=$(stat -c %Y checkpoints/lapa_joint_mgpu/run_seed42/best.pt 2>/dev/null || echo 0)
    if [ "$ody_t" -lt "$tap_t" ] || [ "$jnt_t" -lt "$tap_t" ]; then
      log "Odyssey/Joint checkpoints older than TAPVid; waiting for retrain waves"
      sleep 120
      continue
    fi
    # Avoid racing the primary scheduler: give it 3 minutes to call finish_retrain
    if pgrep -f 'schedule_after_tapvid.sh' >/dev/null 2>&1; then
      log "scheduler still alive; waiting for it to publish"
      sleep 180
      if [ -f "$MARKER" ] || pgrep -f 'finish_retrain.py' >/dev/null 2>&1; then
        log "scheduler/finish in progress or done"
        sleep 60
        continue
      fi
      log "scheduler alive but finish not running; taking over publish"
    fi

    # Skip if Odyssey/Joint look like stale APD=0-only histories from before the fix
    # (new training writes best_apd into ckpt). Require at least one TAPVid best_apd > 5.
    ok=$(python - <<'PY'
import torch
from pathlib import Path
ok=False
for p in Path('checkpoints/lapa_mgpu').glob('run_seed*/best.pt'):
    ckpt=torch.load(p, map_location='cpu')
    if float(ckpt.get('best_apd', 0) or 0) > 5:
        ok=True
print('1' if ok else '0')
PY
)
    if [ "$ok" != "1" ]; then
      log "TAPVid best_apd still too low / missing; not publishing yet"
      sleep 120
      continue
    fi

    log "launching finish_retrain.py"
    if python scripts/finish_retrain.py >> "$LOG" 2>&1; then
      date > "$MARKER"
      log "publish SUCCESS"
      exit 0
    else
      log "publish FAILED (exit $?); will retry in 10 min"
      sleep 600
    fi
  fi
  sleep 120
done
