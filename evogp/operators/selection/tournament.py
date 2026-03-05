import torch
from functools import partial

from .base import BaseSelection


class TournamentSelection(BaseSelection):
    """Tournament selection.

    Individuals from the survival pool compete in random tournaments;
    the winner of each tournament is selected.

    Args:
        tournament_size: Number of contenders per tournament.
        best_probability: Probability that the best contender wins.
            ``1.0`` = always pick the best (deterministic tournament).
        replace: Allow the same individual in multiple tournaments.
        survivor_rate: Fraction of population forming the candidate pool.
    """

    def __init__(
        self,
        tournament_size: int,
        best_probability: float = 1,
        replace: bool = True,
        survivor_rate: float = 1.0,
    ):
        super().__init__(survivor_rate)
        self.t_size = tournament_size
        self.best_p = best_probability
        self.replace = replace

    def _select(self, fitness: torch.Tensor, n: int) -> torch.Tensor:
        total_size = fitness.size(0)
        n_tournament = max(1, total_size // self.t_size)
        k_times = (n - 1) // n_tournament + 1

        @partial(torch.vmap, randomness="different")
        def traverse_once(p):
            return torch.multinomial(
                p, n_tournament * self.t_size, replacement=self.replace
            ).to(torch.int32)

        @torch.vmap
        def t_selection_without_p(contenders):
            contender_fitness = fitness[contenders]
            best_idx = torch.argmax(contender_fitness)[None]
            return contenders[best_idx]

        @partial(torch.vmap, randomness="different")
        def t_selection_with_p(contenders):
            contender_fitness = fitness[contenders]
            idx_rank = torch.argsort(contender_fitness, descending=True)
            random = torch.rand(1).cuda()
            best_p = torch.tensor(self.best_p).cuda()
            nth = (torch.log(random) / torch.log(1 - best_p)).to(torch.int32)
            nth = torch.where(nth >= self.t_size, torch.tensor(0), nth)
            return contenders[idx_rank[nth]]

        p = torch.ones((k_times, total_size), device="cuda")
        contenders = traverse_once(p).reshape(-1, self.t_size)[:n]

        if self.t_size > 1000:
            selected = t_selection_without_p(contenders).reshape(-1)
        else:
            selected = t_selection_with_p(contenders).reshape(-1)

        return selected
