import torch

from .base import Allocator


class EqualWeight(Allocator):
    """w = 1/N. The reference every learned model must justify itself against."""

    def logits(self, X: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return torch.zeros(X.shape[0], self.N, device=X.device), {}
