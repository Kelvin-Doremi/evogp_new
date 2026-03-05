import torch

from evogp.core import CombinedForest
from .base import BaseCrossover


class CombinedDefaultCrossover(BaseCrossover):
    """Random subtree crossover for :class:`CombinedForest`.

    Applies independent random-position crossover to each sub-forest
    while sharing the same (recipient, donor) pairing across all outputs.

    Args:
        crossover_rate: Fraction of pairs that undergo real crossover.
    """

    def _crossover(
        self,
        forest: CombinedForest,
        recipient_indices: torch.Tensor,
        donor_indices: torch.Tensor,
    ) -> CombinedForest:
        n = recipient_indices.shape[0]
        new_forests = []
        for sub_forest in forest.forests:
            tree_sizes = sub_forest.batch_subtree_size[:, 0]
            pos_rand = torch.randint(
                low=0,
                high=torch.iinfo(torch.int32).max,
                size=(2, n),
                dtype=torch.int32,
                device="cuda",
            )
            left_pos = pos_rand[0] % tree_sizes[recipient_indices]
            right_pos = pos_rand[1] % tree_sizes[donor_indices]
            new_forests.append(
                sub_forest.crossover(
                    recipient_indices, donor_indices, left_pos, right_pos
                )
            )
        return CombinedForest(new_forests, forest.data_info)
