# LAPA: Look Around and Pay Attention

**Multi-camera Point Tracking Reimagined with Transformers**

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=github)](https://ostadabbas.github.io/lapa.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.04213-b31b1b?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2512.04213)
[![3DV](https://img.shields.io/badge/3DV-Oral%20Presentation-green?style=for-the-badge)](https://ostadabbas.github.io/lapa.github.io/)

---

## Overview

LAPA is a novel end-to-end transformer-based architecture for multi-camera point tracking. Unlike traditional approaches that separate detection, association, and tracking into distinct stages, LAPA jointly reasons across views and time through attention mechanisms.

**For visualizations, results, and supplementary materials, visit our [Project Page](https://ostadabbas.github.io/lapa.github.io/)**

## Authors

Bishoy Galoaa, Xiangyu Bai, Shayda Moezzi, Utsav Nandi, Sai Siddhartha Vivek Dhir Rangoju, Somaieh Amraee, Sarah Ostadabbas

**Northeastern University**

## Installation

```bash
# Clone this repository
git clone https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-.git
cd Look-Around-and-Pay-Attention-LAPA-

# Install dependencies
pip install -r requirements.txt
```

## Repository Structure

### Core Scripts
- `run_lapa_pipeline.py` - Main LAPA tracking pipeline
- `run_lapa_refinement.py` - Track refinement module
- `run_optimized_pipeline.py` - Optimized version of the pipeline
- `run_all_sequences.py` - Batch processing for multiple sequences

### Training Scripts
- `train_multi_sequence_tracker.py` - Train the multi-sequence tracker
- `train_refinement.py` - Train the refinement module
- `train_with_attention.py` - Train with attention mechanisms

### Evaluation & Analysis
- `evaluate_ablation.py` - Ablation study evaluation
- `run_ablation_study.py` - Run comprehensive ablation studies

### Visualization Tools
- `visualize_attention_weights.py` - Visualize attention weights
- `visualize_trained_attention.py` - Visualize trained attention patterns
- `visualize_refinement_tracks.py` - Visualize refined tracks
- `visualize_fixed_attention.py` - Fixed attention visualization

### Data & Utilities
- `download_tap3d_all_v1.py` - Download TAP-3D dataset
- `lapa/` - Core LAPA implementation package

## Quick Start

### Basic Usage
```bash
# Run LAPA on a single sequence
python run_lapa_pipeline.py --config configs/default.yaml --input_path /path/to/sequence

# Run optimized pipeline
python run_optimized_pipeline.py --sequence your_sequence_name

# Run refinement
python run_lapa_refinement.py --input_tracks tracks.pkl --output refined_tracks.pkl
```

### Training
```bash
# Train the main tracker
python train_multi_sequence_tracker.py --config configs/training.yaml

# Train refinement module
python train_refinement.py --pretrained_model path/to/main/model.pth
```

### Evaluation
```bash
# Run ablation study
python run_ablation_study.py --output_dir results/ablation

# Evaluate specific model
python evaluate_ablation.py --model_path path/to/model.pth --test_data path/to/test/data
```

## Key Features

- **Unified Architecture**: End-to-end transformer-based approach
- **Cross-View Attention**: Geometric-aware attention mechanisms
- **Differentiable Triangulation**: 3D point reconstruction
- **Occlusion Handling**: Robust tracking through occlusions
- **Multi-Camera Support**: Works with arbitrary camera configurations

## Datasets

This implementation supports the TAP-3D dataset. Use the download script:
```bash
python download_tap3d_all_v1.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lapa2025,
  title={LAPA: Look Around and Pay Attention: Multi-camera Point Tracking Reimagined with Transformers},
  author={Galoaa, Bishoy and Bai, Xiangyu and Moezzi, Shayda and Nandi, Utsav and Rangoju, Sai Siddhartha Vivek Dhir and Amraee, Somaieh and Ostadabbas, Sarah},
  journal={arXiv preprint arXiv:2512.04213},
  year={2025}
}
```

## License

This project is released under the MIT License. See the LICENSE file for details.


## Contact

For questions or issues, please open an issue on GitHub.

---

**For complete visualizations and supplementary materials, visit our [Project Page](https://ostadabbas.github.io/lapa.github.io/)**
