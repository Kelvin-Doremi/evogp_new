import torch

from .base import BaseSelection


class RouletteSelection(BaseSelection):
    """Fitness-proportional (roulette wheel) selection within the pool.

    Args:
        survivor_rate: Fraction of population forming the candidate pool.
    """

    def _select(self, fitness: torch.Tensor, n: int) -> torch.Tensor:
        probs = fitness / torch.sum(fitness)
        return torch.multinomial(probs, n, replacement=True).to(torch.int32)
