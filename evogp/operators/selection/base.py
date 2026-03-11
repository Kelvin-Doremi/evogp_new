import torch


class BaseSelection:
    """Base class for selection operators.

    All selections share a common ``survivor_rate`` that restricts the
    candidate pool to the top fraction of the population (sorted by
    fitness descending).  The concrete ``_select`` method then picks *n*
    individuals **within that pool** (with replacement when *n* exceeds
    the pool size).

    Elite preservation is handled at the workflow level
    (:class:`GeneticProgramming`), not here.

    Args:
        survivor_rate: Fraction of the population that forms the
            candidate pool.  ``1.0`` means the entire population is
            eligible; ``0.3`` means only the top 30 %.
    """

    def __init__(self, survivor_rate: float = 1.0):
        assert 0 < survivor_rate <= 1.0, "survivor_rate should be in (0, 1]"
        self.survivor_rate = survivor_rate

    def __call__(self, fitness: torch.Tensor, n: int, **kwargs) -> torch.Tensor:
        """Select *n* individuals from the survival pool.

        Args:
            fitness: Fitness tensor, shape ``(pop_size,)``.
            n: Number of individuals to select.
            **kwargs: Extra info (e.g. ``tree_sizes``) for multi-objective
                selection operators.  Ignored by single-objective selectors.

        Returns:
            Selected indices (into the original population), shape ``(n,)``,
            dtype ``int32``.
        """
        pop_size = fitness.size(0)
        pool_size = max(1, int(pop_size * self.survivor_rate))

        sorted_indices = torch.argsort(fitness, descending=True)
        pool_indices = sorted_indices[:pool_size].to(torch.int32)
        pool_fitness = fitness[pool_indices]

        local_selected = self._select(pool_fitness, n)
        return pool_indices[local_selected]

    def _select(self, fitness: torch.Tensor, n: int) -> torch.Tensor:
        """Core selection logic — pick *n* indices from the pool.

        Args:
            fitness: Fitness of the survival pool in **descending** order,
                shape ``(pool_size,)``.
            n: Number to select.  May exceed ``pool_size``; implementations
                should handle this (typically via replacement).

        Returns:
            Local indices within the pool, shape ``(n,)``, dtype ``int32``.
        """
        raise NotImplementedError
