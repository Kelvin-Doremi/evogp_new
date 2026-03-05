import torch

from .base import BaseSelection


class DefaultSelection(BaseSelection):
    """Deterministic top-*n* selection.

    Always picks the best individuals from the survival pool.  When *n*
    exceeds the pool size, the pool is cycled (the best individual
    appears most often).
    """

    def _select(self, fitness: torch.Tensor, n: int) -> torch.Tensor:
        pool_size = fitness.size(0)
        return torch.arange(n, device=fitness.device, dtype=torch.int32) % pool_size
