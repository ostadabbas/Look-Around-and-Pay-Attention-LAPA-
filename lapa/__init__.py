"""
LAPA (Look Around and Pay Attention) - Multi-Camera Point Tracking.

Paper: https://arxiv.org/abs/2512.04213
"""

__version__ = "0.1.0"

from lapa.models.lapa import LAPA, count_parameters, build_w2c_normalized

__all__ = ["LAPA", "count_parameters", "build_w2c_normalized", "__version__"]
