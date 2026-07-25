from .attention import CrossSectionalAttention
from .equal_weight import EqualWeight
from .linear import LinearAllocator
from .p7 import CorrBiasAttention, GatedCrossSectional, P7Encoder

REGISTRY = {
    "equal_weight": EqualWeight,
    "linear": LinearAllocator,
    "attention": CrossSectionalAttention,
    "p7_encoder": P7Encoder,
    "gated_attention": GatedCrossSectional,
    "corr_bias_attention": CorrBiasAttention,
}
