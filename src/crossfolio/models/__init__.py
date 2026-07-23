from .attention import CrossSectionalAttention
from .equal_weight import EqualWeight
from .linear import LinearAllocator

REGISTRY = {
    "equal_weight": EqualWeight,
    "linear": LinearAllocator,
    "attention": CrossSectionalAttention,
}
