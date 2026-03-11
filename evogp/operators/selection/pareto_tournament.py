"""
Pareto Tournament Selection.

Two objectives: fitness (higher = better) and tree size (lower = better).

For each tournament of k individuals:
  1. Pairwise Pareto dominance is checked.
  2. Among non-dominated contenders:
     - k <= 3: random selection (too few individuals for meaningful CD).
     - k >  3: the one with the highest crowding distance wins.

Crowding distance (when used) is precomputed once for the entire
candidate pool.  Unlike NSGA-II (which computes CD within each
non-domination front and assigns infinity to boundary points), here CD
is computed over the whole pool.  Boundary individuals use one-sided
extrapolation instead of infinity, so a bad solution that happens to sit
at an extreme position does not get an undeserved advantage.

All tournaments are fully vectorized — no Python loops over tournaments.
"""
import torch

from .base import BaseSelection


class ParetoTournamentSelection(BaseSelection):
    """Pareto Tournament Selection.

    For each of *n* parent slots:
      1. Sample *k* random contenders from the survival pool.
      2. Check pairwise Pareto dominance (O(k²), negligible for k ≤ 7).
      3. Among non-dominated contenders:
         - *k ≤ 3*: pick one uniformly at random.
         - *k > 3*: pick the one with the highest crowding distance.

    When ``tree_sizes`` is not provided, falls back to a standard
    single-objective tournament on fitness alone.

    Args:
        tournament_size: Number of contenders per tournament (k).
        survivor_rate: Fraction of population forming the candidate pool.
    """

    def __init__(
        self,
        tournament_size: int = 3,
        survivor_rate: float = 1.0,
    ):
        super().__init__(survivor_rate)
        self.t_size = tournament_size

    # ------------------------------------------------------------------
    # Override __call__ to intercept tree_sizes
    # ------------------------------------------------------------------

    def __call__(self, fitness: torch.Tensor, n: int, **kwargs) -> torch.Tensor:
        tree_sizes = kwargs.get("tree_sizes", None)

        if tree_sizes is None:
            return super().__call__(fitness, n)

        pop_size = fitness.size(0)
        pool_size = max(1, int(pop_size * self.survivor_rate))

        if pool_size < pop_size:
            sorted_indices = torch.argsort(fitness, descending=True)
            pool_indices = sorted_indices[:pool_size].to(torch.int32)
            pool_fitness = fitness[pool_indices]
            pool_sizes = tree_sizes[pool_indices]
        else:
            pool_indices = torch.arange(
                pop_size, device=fitness.device, dtype=torch.int32)
            pool_fitness = fitness
            pool_sizes = tree_sizes

        pool_cd = (
            self._crowding_distance(pool_fitness, pool_sizes)
            if self.t_size > 3 else None
        )
        local_selected = self._pareto_select(
            pool_fitness, pool_sizes, pool_cd, n)
        return pool_indices[local_selected]

    # ------------------------------------------------------------------
    # Crowding distance (2 objectives, pool-level)
    # ------------------------------------------------------------------

    @staticmethod
    def _crowding_distance(
        fitness: torch.Tensor, tree_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute crowding distance for every individual in the pool.

        For each objective the individuals are sorted:
          - Interior (rank 1 … N-2):
                ``(val[rank+1] − val[rank−1]) / range``
          - Boundaries (rank 0 and N-1):
                doubled one-sided gap, i.e.
                ``2 * (val[1] − val[0]) / range`` for rank 0, etc.

        Returns:
            Tensor of shape ``(pool_size,)`` — higher means more isolated.
        """
        pool_size = fitness.size(0)
        device = fitness.device

        if pool_size <= 2:
            return torch.ones(pool_size, device=device)

        cd = torch.zeros(pool_size, device=device)

        for obj in (fitness, tree_sizes.float()):
            sorted_idx = torch.argsort(obj)
            sorted_vals = obj[sorted_idx]

            obj_range = sorted_vals[-1] - sorted_vals[0]
            if obj_range.abs() < 1e-12:
                continue

            inv_range = 1.0 / obj_range

            # boundaries: doubled one-sided distance
            cd[sorted_idx[0]] += (2.0 * (sorted_vals[1] - sorted_vals[0]) * inv_range).abs()
            cd[sorted_idx[-1]] += (2.0 * (sorted_vals[-1] - sorted_vals[-2]) * inv_range).abs()

            # interior: symmetric two-sided distance
            neighbour_diff = (sorted_vals[2:] - sorted_vals[:-2]) * inv_range
            cd[sorted_idx[1:-1]] += neighbour_diff.abs()

        return cd

    # ------------------------------------------------------------------
    # Core: vectorized Pareto tournament with CD tiebreak
    # ------------------------------------------------------------------

    def _pareto_select(
        self,
        fitness: torch.Tensor,
        tree_sizes: torch.Tensor,
        crowding_dist: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        """Fully-vectorized Pareto tournament.

        Args:
            fitness:       Pool fitness, shape ``(pool,)``, higher = better.
            tree_sizes:    Pool tree sizes, shape ``(pool,)``, lower = better.
            crowding_dist: Precomputed CD, shape ``(pool,)``.
            n:             Number of winners to produce.

        Returns:
            Selected local indices, shape ``(n,)``, dtype ``int32``.
        """
        pool_size = fitness.size(0)
        device = fitness.device
        k = self.t_size

        # --- 1. Draw tournament groups: (n, k) ────────────────────────
        contenders = torch.randint(0, pool_size, (n, k), device=device)

        cont_fit = fitness[contenders]
        cont_size = tree_sizes[contenders].float()

        # --- 2. Pairwise dominance: (n, k, k) ─────────────────────────
        fit_i = cont_fit.unsqueeze(2)    # (n, k, 1)
        fit_j = cont_fit.unsqueeze(1)    # (n, 1, k)
        sz_i = cont_size.unsqueeze(2)
        sz_j = cont_size.unsqueeze(1)

        dominates = (
            (fit_i >= fit_j) & (sz_i <= sz_j)
            & ((fit_i > fit_j) | (sz_i < sz_j))
        )

        # --- 3. Non-dominated mask ────────────────────────────────────
        is_dominated = dominates.any(dim=1)          # (n, k)
        non_dominated = ~is_dominated                 # (n, k)

        # --- 4. Tiebreak among non-dominated contenders ───────────────
        if crowding_dist is not None:
            cont_cd = crowding_dist[contenders]       # (n, k)
            scores = torch.where(
                non_dominated,
                cont_cd + torch.rand(n, k, device=device) * 1e-4,
                torch.tensor(-1.0, device=device),
            )
        else:
            scores = torch.where(
                non_dominated,
                torch.rand(n, k, device=device),
                torch.tensor(-1.0, device=device),
            )
        winner_local = torch.argmax(scores, dim=1)    # (n,)

        return contenders[
            torch.arange(n, device=device), winner_local
        ].to(torch.int32)

    # ------------------------------------------------------------------
    # Fallback: single-objective tournament (no tree_sizes)
    # ------------------------------------------------------------------

    def _select(self, fitness: torch.Tensor, n: int) -> torch.Tensor:
        """Standard fitness-only tournament (fallback)."""
        pool_size = fitness.size(0)
        device = fitness.device
        k = self.t_size

        contenders = torch.randint(0, pool_size, (n, k), device=device)
        cont_fit = fitness[contenders]
        winner_local = torch.argmax(cont_fit, dim=1)

        return contenders[
            torch.arange(n, device=device), winner_local
        ].to(torch.int32)
