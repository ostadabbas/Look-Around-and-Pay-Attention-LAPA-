#!/usr/bin/env bash
# Train LAPA jointly on TAPVid-3D-MC + PointOdyssey-MC (single GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ -f /opt/anaconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/anaconda3/etc/profile.d/conda.sh
  conda activate lapa
fi

GPU="${GPU:-1}"
OUT="${OUT:-checkpoints/lapa_joint}"
PY="${PYTHON:-python}"
mkdir -p "$OUT" outputs/runs_joint

echo "Launching joint train on physical GPU $GPU -> $OUT"
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
    --seed 42
