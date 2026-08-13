#!/usr/bin/env bash
# Launch 4 independent joint (TAPVid + PointOdyssey) LAPA trainings.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ -f /opt/anaconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/anaconda3/etc/profile.d/conda.sh
  conda activate lapa
fi

mkdir -p checkpoints/lapa_joint_mgpu outputs/runs_joint
PY="${PYTHON:-/home/bi.ga/.conda/envs/lapa/bin/python}"

SEEDS=(42 123 7 99)
for i in 0 1 2 3; do
  GPU=$((i + 1))
  SEED=${SEEDS[$i]}
  OUT="checkpoints/lapa_joint_mgpu/run_seed${SEED}"
  mkdir -p "$OUT"
  echo "Launching joint seed=$SEED on physical GPU $GPU -> $OUT"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU "$PY" train_lapa.py \
      --dataset joint \
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
      --tapvid_prob 0.5 \
      --seed "$SEED" \
      > "outputs/runs_joint/seed${SEED}.log" 2>&1 &
  echo $! > "outputs/runs_joint/seed${SEED}.pid"
done

echo "All 4 joint jobs launched. PIDs:"
cat outputs/runs_joint/seed*.pid
echo "Monitor with: tail -f outputs/runs_joint/seed42.log"
