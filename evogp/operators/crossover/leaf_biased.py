import torch
from torch import Tensor

from evogp.core import Forest, randint
from .base import BaseCrossover


class LeafBiasedCrossover(BaseCrossover):
    """Subtree crossover biased towards leaf nodes.

    With probability ``leaf_bias`` the crossover positions are restricted
    to leaf nodes (subtree_size == 1); otherwise random positions are used
    like :class:`DefaultCrossover`.

    Args:
        leaf_bias: Probability of choosing leaf-node positions.
        crossover_rate: Fraction of pairs that undergo real crossover.
    """

    def __init__(self, leaf_bias: float = 0.3, crossover_rate: float = 1.0):
        super().__init__(crossover_rate)
        self.leaf_bias = leaf_bias

    def _crossover(
        self,
        forest: Forest,
        recipient_indices: torch.Tensor,
        donor_indices: torch.Tensor,
    ) -> Forest:
        n = recipient_indices.shape[0]
        size_tensor = forest.batch_subtree_size

        def _choose_leaf_pos(subtree_sizes: Tensor) -> Tensor:
            random = torch.rand(subtree_sizes.shape, device="cuda")
            arange = torch.arange(subtree_sizes.shape[1], device="cuda")
            mask = arange < subtree_sizes[:, 0].unsqueeze(1)
            random = random * mask
            random = torch.where(subtree_sizes == 1, random, 0)
            return torch.argmax(random, 1).to(torch.int32)

        recipient_leaf_pos = _choose_leaf_pos(size_tensor[recipient_indices])
        donor_leaf_pos = _choose_leaf_pos(size_tensor[donor_indices])

        recipient_normal_pos = randint(
            size=(n,), low=0, high=size_tensor[recipient_indices, 0], dtype=torch.int32
        )
        donor_normal_pos = randint(
            size=(n,), low=0, high=size_tensor[donor_indices, 0], dtype=torch.int32
        )

        leaf_pair = torch.rand(n, device="cuda") < self.leaf_bias
        recipient_pos = torch.where(leaf_pair, recipient_leaf_pos, recipient_normal_pos)
        donor_pos = torch.where(leaf_pair, donor_leaf_pos, donor_normal_pos)

        return forest.crossover(
            recipient_indices, donor_indices, recipient_pos, donor_pos
        )
