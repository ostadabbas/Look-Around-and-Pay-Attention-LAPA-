# LAPA: Look Around and Pay Attention

**Multi-camera Point Tracking Reimagined with Transformers**

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=github)](https://ostadabbas.github.io/lapa.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.04213-b31b1b?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2512.04213)
[![3DV](https://img.shields.io/badge/3DV-Oral%20Presentation-green?style=for-the-badge)](https://ostadabbas.github.io/lapa.github.io/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Pretrained%20Weights-yellow?style=for-the-badge)](https://huggingface.co/bishoygaloaa/lapa-tapvid3d-mc)
[![GitHub Release](https://img.shields.io/badge/GitHub-Download%20Weights-black?style=for-the-badge&logo=github)](https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-/releases/tag/lapa-weights)

---

## Overview

LAPA is an end-to-end transformer-based architecture for multi-camera point tracking. It jointly reasons across views and time through distance-based volumetric attention, producing consistent 3D trajectories without classical triangulation.

## Authors

Bishoy Galoaa, Xiangyu Bai, Shayda Moezzi, Utsav Nandi, Sai Siddhartha Vivek Dhir Rangoju, Somaieh Amraee, Sarah Ostadabbas

**Northeastern University**

## Installation

```bash
git clone https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-.git
cd Look-Around-and-Pay-Attention-LAPA-
pip install -r requirements.txt
```

## Pretrained Weights

| Dataset | Hugging Face | GitHub Release |
|---------|--------------|----------------|
| TAPVid-3D-MC | [bishoygaloaa/lapa-tapvid3d-mc](https://huggingface.co/bishoygaloaa/lapa-tapvid3d-mc) | [lapa-weights](https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-/releases/tag/lapa-weights) (`lapa.pt`) |
| PointOdyssey-MC | Coming soon (`bishoygaloaa/lapa-pointodyssey-mc`) | Coming soon |

**Direct download (TAPVid-3D-MC):**
- Hugging Face: https://huggingface.co/bishoygaloaa/lapa-tapvid3d-mc/resolve/main/lapa.pt
- GitHub: https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-/releases/download/lapa-weights/lapa.pt

```bash
# Hugging Face CLI
hf download bishoygaloaa/lapa-tapvid3d-mc lapa.pt --local-dir checkpoints/lapa

# Or Python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("bishoygaloaa/lapa-tapvid3d-mc", "lapa.pt")
```

## Quick Start

### Inference on a 3-camera scene

```bash
python inference_lapa.py \
  --checkpoint checkpoints/lapa/best.pt \
  --scene boxes --cameras 5 6 7 \
  --feature_dir data/feature_cache \
  --output outputs/inference_boxes.npz
```

### Evaluate

```bash
python evaluate_lapa.py \
  --checkpoint checkpoints/lapa/best.pt \
  --output outputs/eval_metrics.json
```

## Data: TAPVid-3D-MC

```bash
# 1) Download TAPVid-3D pstudio (minival) + Dynamic3DGaussians calibration/frames
python scripts/download_data.py --output_dir ./data

# 2) Full-eval annotations are at v1.0/pstudio (joined with D3G images by the
#    helper in scripts/download_data.py / the session build script)

# 3) Build multi-camera metadata + world tracks
python -m lapa.data.mc_builder \
  --npz_root ./data --d3g_root ./data/d3g/data --out_dir ./data/tapvid3d_mc

# 4) Precompute DINOv2 features (and optional CoTracker tracks)
python -m lapa.features.precompute --device cuda:0 --max_points 256
```

## Training

```bash
# Single GPU
python train_lapa.py --device cuda:0 --output_dir checkpoints/lapa --epochs 50

# 4× GPU (one job per V100)
bash scripts/launch_4gpu_train.sh
```

Paper recipe: AdamW lr=1e-4, wd=1e-5, cosine + 5-epoch warmup,  
\(\mathcal{L}=1.0\mathcal{L}_{recon}+0.7\mathcal{L}_{proj}+0.8\mathcal{L}_{attn}\).

## Repository Layout

| Path | Purpose |
|------|---------|
| `lapa/models/lapa.py` | Paper method (volumetric attention + triangulation MLP) |
| `lapa/data/mc_builder.py` | TAPVid-3D-MC construction + calibration gate |
| `lapa/data/mc_dataset.py` | 3-camera triplet dataset |
| `lapa/features/precompute.py` | DINOv2 (+ optional CoTracker) cache |
| `lapa/losses.py` | Multi-objective loss |
| `lapa/eval/metrics.py` | Official TAPVid-3D metrics |
| `train_lapa.py` | Training entry point |
| `evaluate_lapa.py` | Evaluation |
| `inference_lapa.py` | Inference |

Legacy prototype modules under `lapa/models/geometric_attention*.py` are kept for reference but are **not** used by the release pipeline.

## Citation

```bibtex
@article{lapa2025,
  title={LAPA: Look Around and Pay Attention: Multi-camera Point Tracking Reimagined with Transformers},
  author={Galoaa, Bishoy and Bai, Xiangyu and Moezzi, Shayda and Nandi, Utsav and Rangoju, Sai Siddhartha Vivek Dhir and Amraee, Somaieh and Ostadabbas, Sarah},
  journal={arXiv preprint arXiv:2512.04213},
  year={2025}
}
```

## License

This project is released under the MIT License. Respect TAPVid-3D / Panoptic Studio / Dynamic3DGaussians licenses when using the data.

## Contact

For questions or issues, please open an issue on GitHub.

---

**For visualizations and supplementary materials, visit our [Project Page](https://ostadabbas.github.io/lapa.github.io/)**
