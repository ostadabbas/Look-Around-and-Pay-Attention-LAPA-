#!/usr/bin/env bash
# Wait for CoTracker cache recompute, verify DLT baseline, then retrain.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lapa
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$ROOT"
mkdir -p outputs

echo "[$(date)] waiting for precompute jobs..."
while pgrep -f 'python lapa/features/precompute_canonical.py' >/dev/null; do
  eval_n=$(find data/feature_cache_eval -name '*.h5' 2>/dev/null | wc -l)
  canon_n=$(find data/feature_cache_canonical -name '*.h5' 2>/dev/null | wc -l)
  echo "[$(date)] eval_h5=$eval_n canon_h5=$canon_n"
  sleep 60
done

echo "[$(date)] precompute finished"
eval_n=$(find data/feature_cache_eval -name '*.h5' 2>/dev/null | wc -l)
canon_n=$(find data/feature_cache_canonical -name '*.h5' 2>/dev/null | wc -l)
echo "eval_h5=$eval_n canon_h5=$canon_n"
tail -n 20 outputs/precompute/eval.log || true
tail -n 10 outputs/precompute/canon_a.log || true

echo "[$(date)] running CoTracker DLT baseline..."
python scripts/triangulation_baseline.py --max_points 256 \
  --output outputs/dlt_baseline_cotracker.json \
  > outputs/dlt_cotracker.log 2>&1
python - <<'PY'
import json
d=json.load(open('outputs/dlt_baseline_cotracker.json'))
print('CoTracker DLT:', {k:d[k] for k in ['APD','OA','AJ3D','AJ2D','n_samples']})
if d['APD'] < 20:
    print('WARNING: CoTracker DLT APD still low; continuing but investigate.')
PY

echo "[$(date)] launching TAPVid 4-seed retrain..."
rm -rf checkpoints/lapa_mgpu/run_seed*
bash scripts/launch_4gpu_train.sh
nohup bash scripts/schedule_after_tapvid.sh > outputs/schedule_retrain.log 2>&1 &
echo $! > outputs/schedule_retrain.pid
echo "[$(date)] schedule pid $(cat outputs/schedule_retrain.pid)"
