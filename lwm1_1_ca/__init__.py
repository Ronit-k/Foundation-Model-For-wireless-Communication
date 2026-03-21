# =============================================================================
# lwm1_1_ca/__init__.py  —  Package exports
#
# This package implements Coordinate Attention (CA) augmented pretraining
# for Large Wireless Model v1.1 (LWM 1.1).
# =============================================================================

from lwm1_1_ca.coordatt import CoordAtt, HSwish, HSigmoid
from lwm1_1_ca.torch_pipeline_v11 import (
    LWM11WithPrepatchCA,
    ensure_ri_channels,
    add_complex_noise_ri,
    channels_to_patches_v11,
    mask_patches_v11,
)

__version__ = "1.0.0"

__all__ = [
    # Coordinate Attention
    "CoordAtt",
    "HSigmoid",
    "HSwish",
    # End-to-end pretraining model
    "LWM11WithPrepatchCA",
    # Pipeline helpers (useful for downstream evaluation)
    "ensure_ri_channels",
    "add_complex_noise_ri",
    "channels_to_patches_v11",
    "mask_patches_v11",
]
