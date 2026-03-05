import torch

from .base import BaseSelection


class RankSelection(BaseSelection):
    r"""Rank-based selection with configurable selection pressure.

    Selection probability for rank *i* (0 = best) inside the pool:

    .. math::
        P(R_i) = \frac{1}{n}\left(1 + sp\left(1 - \frac{2i}{n-1}\right)\right)

    Args:
        selection_pressure: Value in ``[0, 1]``.  Higher → stronger bias
            towards top-ranked individuals.
        survivor_rate: Fraction of population forming the candidate pool.
    """

    def __init__(self, selection_pressure: float = 0.5, survivor_rate: float = 1.0):
        super().__init__(survivor_rate)
        assert 0 <= selection_pressure <= 1
        self.sp = selection_pressure

    def _select(self, fitness: torch.Tensor, n: int) -> torch.Tensor:
        pool_size = fitness.size(0)
        ranks = torch.arange(pool_size, device="cuda", dtype=torch.float32)
        denom = max(pool_size - 1, 1)
        probs = (1 / pool_size) * (1 + self.sp * (1 - 2 * ranks / denom))
        return torch.multinomial(probs, n, replacement=True).to(torch.int32)
