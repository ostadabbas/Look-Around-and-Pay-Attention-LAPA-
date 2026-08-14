#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lapa
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$ROOT"
mkdir -p outputs/precompute data/feature_cache_eval data/feature_cache_canonical

pkill -f 'precompute_canonical.py' 2>/dev/null || true
sleep 1

nohup python lapa/features/precompute_canonical.py --mode eval --use_cotracker --device cuda:1 --max_points 256 \
  > outputs/precompute/eval.log 2>&1 &
echo $! > outputs/precompute/eval.pid

nohup python lapa/features/precompute_canonical.py --mode canonical --use_cotracker --device cuda:2 --scenes basketball boxes \
  > outputs/precompute/canon_a.log 2>&1 &
echo $! > outputs/precompute/canon_a.pid

nohup python lapa/features/precompute_canonical.py --mode canonical --use_cotracker --device cuda:3 --scenes football juggle \
  > outputs/precompute/canon_b.log 2>&1 &
echo $! > outputs/precompute/canon_b.pid

nohup python lapa/features/precompute_canonical.py --mode canonical --use_cotracker --device cuda:4 --scenes softball tennis \
  > outputs/precompute/canon_c.log 2>&1 &
echo $! > outputs/precompute/canon_c.pid

echo "launched:"
cat outputs/precompute/*.pid
sleep 15
pgrep -af 'precompute_canonical.py' || true
tail -n 5 outputs/precompute/eval.log || true
tail -n 3 outputs/precompute/canon_a.log || true
