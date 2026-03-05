import torch

from evogp.core import Forest
from .base import BaseCrossover


class DefaultCrossover(BaseCrossover):
    """Random subtree crossover.

    For each (recipient, donor) pair, a random position is chosen in both
    trees and the recipient's subtree at that position is replaced with
    the donor's subtree.

    Args:
        crossover_rate: Fraction of pairs that undergo real crossover.
    """

    def _crossover(
        self,
        forest: Forest,
        recipient_indices: torch.Tensor,
        donor_indices: torch.Tensor,
    ) -> Forest:
        n = recipient_indices.shape[0]
        tree_sizes = forest.batch_subtree_size[:, 0]

        pos_rand = torch.randint(
            low=0,
            high=torch.iinfo(torch.int32).max,
            size=(2, n),
            dtype=torch.int32,
            device="cuda",
        )
        recipient_pos = pos_rand[0] % tree_sizes[recipient_indices]
        donor_pos = pos_rand[1] % tree_sizes[donor_indices]

        return forest.crossover(
            recipient_indices, donor_indices, recipient_pos, donor_pos
        )
