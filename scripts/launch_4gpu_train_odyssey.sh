#!/usr/bin/env bash
# Launch 4 independent PointOdyssey-MC trainings (one per V100).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lapa

mkdir -p checkpoints/lapa_odyssey_mgpu outputs/runs_odyssey

SEEDS=(42 123 7 99)
for i in 0 1 2 3; do
  GPU=$((i + 1))
  SEED=${SEEDS[$i]}
  OUT="checkpoints/lapa_odyssey_mgpu/run_seed${SEED}"
  mkdir -p "$OUT"
  echo "Launching Odyssey seed=$SEED on physical GPU $GPU -> $OUT"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU nohup python train_lapa.py \
      --dataset pointodyssey \
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
      > "outputs/runs_odyssey/seed${SEED}.log" 2>&1 &
  echo $! > "outputs/runs_odyssey/seed${SEED}.pid"
done

echo "All 4 Odyssey jobs launched. PIDs:"
cat outputs/runs_odyssey/seed*.pid
echo "Monitor with: tail -f outputs/runs_odyssey/seed42.log"
