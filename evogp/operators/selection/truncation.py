import torch

from .base import BaseSelection


class TruncationSelection(BaseSelection):
    """Uniform random selection from the survival pool.

    Equivalent to truncation selection: the top ``survivor_rate`` fraction
    of the population is kept, and *n* individuals are drawn uniformly at
    random (with replacement) from that pool.

    Args:
        survivor_rate: Fraction of population forming the candidate pool.
    """

    def _select(self, fitness: torch.Tensor, n: int) -> torch.Tensor:
        pool_size = fitness.size(0)
        return torch.randint(0, pool_size, (n,), device="cuda", dtype=torch.int32)
