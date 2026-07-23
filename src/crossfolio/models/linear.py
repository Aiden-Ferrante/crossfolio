import torch
import torch.nn as nn

from .base import Allocator


class LinearAllocator(Allocator):
    """flatten(N*T) -> Linear -> N logits. Deliberately zero hidden layers: the
    bag-of-words baseline. Params grow with N*T*N (~864k at N=120) against a few
    hundred effective independent months — expected to overfit; that's the lesson.
    """

    def __init__(self, N, T, cfg):
        super().__init__(N, T, cfg)
        self.fc = nn.Linear(N * T, N)

    def logits(self, X: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return self.fc(X.flatten(1)), {}
