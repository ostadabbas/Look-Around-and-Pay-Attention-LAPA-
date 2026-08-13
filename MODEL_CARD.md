# LAPA

Pretrained weights for [LAPA](https://arxiv.org/abs/2512.04213) (Look Around and Pay Attention) — multi-camera point tracking with transformers.

## Load

```python
import torch
from lapa.models.lapa import LAPA

ckpt = torch.load("lapa.pt", map_location="cpu")
model = LAPA()
model.load_state_dict(ckpt["model"])
model.eval()
```

Or with the repo helper:

```bash
python inference_lapa.py --checkpoint lapa.pt --scene boxes --cameras 5 6 7
```

## Training data

TAPVid-3D Panoptic Studio split, extended to multi-camera (TAPVid-3D-MC) using Dynamic3DGaussians calibration.

## Citation

```bibtex
@article{lapa2025,
  title={LAPA: Look Around and Pay Attention: Multi-camera Point Tracking Reimagined with Transformers},
  author={Galoaa, Bishoy and Bai, Xiangyu and Moezzi, Shayda and Nandi, Utsav and Rangoju, Sai Siddhartha Vivek Dhir and Amraee, Somaieh and Ostadabbas, Sarah},
  journal={arXiv preprint arXiv:2512.04213},
  year={2025}
}
```

## Links

- Paper: https://arxiv.org/abs/2512.04213
- Code: https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-
