import torch
from evogp.core import Forest


class BaseCrossover:
    """Base class for crossover operators.

    All crossover operators share a common ``crossover_rate`` that
    controls what fraction of (recipient, donor) pairs actually undergo
    crossover.  The remaining pairs simply return the recipient tree
    unchanged.

    Subclasses implement ``_crossover`` with the core crossover logic;
    the base ``__call__`` handles the rate-based split automatically.

    Args:
        crossover_rate: Fraction of pairs that undergo real crossover.
            ``1.0`` = all pairs are crossed; ``0.8`` = 80 % crossed,
            20 % just copy the recipient.
    """

    def __init__(self, crossover_rate: float = 1.0):
        assert 0 <= crossover_rate <= 1.0, "crossover_rate should be in [0, 1]"
        self.crossover_rate = crossover_rate

    def __call__(
        self,
        forest: Forest,
        recipient_indices: torch.Tensor,
        donor_indices: torch.Tensor,
    ) -> Forest:
        n = recipient_indices.shape[0]
        crossover_cnt = int(n * self.crossover_rate)

        if crossover_cnt >= n:
            return self._crossover(forest, recipient_indices, donor_indices)
        if crossover_cnt <= 0:
            return forest[recipient_indices]

        perm = torch.randperm(n, device=recipient_indices.device)
        cx_idx = perm[:crossover_cnt]
        no_cx_idx = perm[crossover_cnt:]

        crossed = self._crossover(
            forest, recipient_indices[cx_idx], donor_indices[cx_idx]
        )
        static = forest[recipient_indices[no_cx_idx]]
        return crossed + static

    def _crossover(
        self,
        forest: Forest,
        recipient_indices: torch.Tensor,
        donor_indices: torch.Tensor,
    ) -> Forest:
        """Core crossover logic — subclasses implement this.

        Args:
            forest: The entire population forest.
            recipient_indices: Indices of recipients to cross, shape ``(m,)``.
            donor_indices: Indices of donors to cross, shape ``(m,)``.

        Returns:
            New forest of ``m`` crossed offspring.
        """
        raise NotImplementedError
