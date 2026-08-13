#!/usr/bin/env bash
# Launch 4 independent LAPA trainings (one per V100), then pick the best.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lapa

mkdir -p checkpoints/lapa_mgpu outputs/runs

# Physical GPUs 1-4 are V100s (0 is GTX 745)
SEEDS=(42 123 7 99)
for i in 0 1 2 3; do
  GPU=$((i + 1))
  SEED=${SEEDS[$i]}
  OUT="checkpoints/lapa_mgpu/run_seed${SEED}"
  mkdir -p "$OUT"
  echo "Launching seed=$SEED on physical GPU $GPU -> $OUT"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU python train_lapa.py \
      --device cuda:0 \
      --output_dir "$OUT" \
      --epochs 50 \
      --warmup_epochs 5 \
      --lr 1e-4 \
      --steps_per_epoch 250 \
      --val_steps 40 \
      --num_frames 24 \
      --max_points 64 \
      --num_views 3 \
      --seed "$SEED" \
      > "outputs/runs/seed${SEED}.log" 2>&1 &
  echo $! > "outputs/runs/seed${SEED}.pid"
done

echo "All 4 jobs launched. PIDs:"
cat outputs/runs/seed*.pid
echo "Monitor with: tail -f outputs/runs/seed42.log"
