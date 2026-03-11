"""
DE (Differential Evolution) constant optimization for GP trees.

Ported from evox DE.  Uses a batched implementation: all optimized
individuals run as one set of tensor operations (pad to max_dim),
with a single batched SR_fitness call per iteration for evaluation.

Supports DE/rand/1/bin and DE/best/1/bin strategies.
"""
from typing import List, Tuple

import numpy as np
import torch

from .base import BaseOptimization
from .cmaes import _BatchForestBuilder, _to_device, _extract_constants, _write_constants
from evogp.core import Forest


class BatchedDE:
    """N DE instances batched into (N, ...) tensors.

    Each instance maintains a population of ``pop_size`` individuals and
    evolves them via differential mutation + binomial crossover with greedy
    selection (lower MSE wins).
    """

    def __init__(
        self,
        x0_list: List[np.ndarray],
        pop_size: int,
        differential_weight: float = 0.5,
        cross_probability: float = 0.9,
        base_vector: str = "rand",
        init_stdev: float = 1.0,
        device: torch.device = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        N = len(x0_list)
        dims = [len(x0) for x0 in x0_list]
        D = max(dims) if dims else 1
        self.N, self.D, self.pop_size = N, D, pop_size
        self.dims = torch.tensor(dims, device=device)
        self.F = differential_weight
        self.CR = cross_probability
        self.use_best = (base_vector == "best")

        self.dim_mask = torch.zeros(N, D, device=device)
        for i, d in enumerate(dims):
            self.dim_mask[i, :d] = 1.0

        center = torch.zeros(N, D, device=device)
        for i, x0 in enumerate(x0_list):
            center[i, : len(x0)] = torch.tensor(x0, dtype=torch.float32)

        noise = torch.randn(N, pop_size, D, device=device) * init_stdev
        self.pop = (center.unsqueeze(1) + noise) * self.dim_mask.unsqueeze(1)
        self.pop[:, 0, :] = center

        self.pop_fitness = torch.full((N, pop_size), float("inf"), device=device)
        self._initialized = False

        self._best_x = [np.array(x0, dtype=np.float64) for x0 in x0_list]
        self._best_mse = torch.full((N,), float("inf"), device=device)

    # ── ask / tell ──────────────────────────────────────────────

    def ask(self) -> torch.Tensor:
        """Return (N, pop_size, D) candidates for evaluation."""
        if not self._initialized:
            return self.pop.clone()

        N, ps, D = self.N, self.pop_size, self.D
        device = self.device

        if self.use_best:
            best_idx = torch.argmin(self.pop_fitness, dim=1)
            base = self.pop[torch.arange(N, device=device), best_idx]
            base = base.unsqueeze(1).expand(-1, ps, -1)
        else:
            r0 = torch.randint(0, ps, (N, ps), device=device)
            base = torch.gather(self.pop, 1, r0.unsqueeze(-1).expand(-1, -1, D))

        r1 = torch.randint(0, ps, (N, ps), device=device)
        r2 = torch.randint(0, ps, (N, ps), device=device)
        v1 = torch.gather(self.pop, 1, r1.unsqueeze(-1).expand(-1, -1, D))
        v2 = torch.gather(self.pop, 1, r2.unsqueeze(-1).expand(-1, -1, D))

        mutant = base + self.F * (v1 - v2)

        cross_mask = torch.rand(N, ps, D, device=device) < self.CR
        j_rand = torch.randint(0, D, (N, ps, 1), device=device)
        cross_mask.scatter_(2, j_rand, True)

        trial = torch.where(cross_mask, mutant, self.pop)
        return trial * self.dim_mask.unsqueeze(1)

    def tell(self, populations: torch.Tensor, mse: torch.Tensor):
        N = self.N
        if not self._initialized:
            self.pop_fitness = mse.clone()
            self._initialized = True
        else:
            better = mse < self.pop_fitness
            self.pop = torch.where(better.unsqueeze(-1), populations, self.pop)
            self.pop_fitness = torch.where(better, mse, self.pop_fitness)

        best_idx = torch.argmin(self.pop_fitness, dim=1)
        best_mse = self.pop_fitness[torch.arange(N, device=self.device), best_idx]
        improved = best_mse < self._best_mse
        if improved.any():
            self._best_mse = torch.where(improved, best_mse, self._best_mse)
            for i in torch.where(improved)[0].tolist():
                d = int(self.dims[i])
                self._best_x[i] = (
                    self.pop[i, best_idx[i], :d].cpu().numpy().astype(np.float64)
                )

    def get_best(self, i: int) -> Tuple[np.ndarray, float]:
        return self._best_x[i], float(self._best_mse[i])


# ═══════════════════════════════════════════════════════════════════
# Public operator
# ═══════════════════════════════════════════════════════════════════

class DEOptimization(BaseOptimization):
    """Optimize constants in GP trees using batched DE.

    Args:
        n_optimize:           How many top individuals to optimize.
        n_offspring:          DE population size per individual.
        opt_iterations:       Number of DE generations.
        differential_weight:  Scale factor F.
        cross_probability:    Crossover rate CR.
        base_vector:          "rand" or "best".
        init_stdev:           Std of initial population around x0.
    """

    def __init__(
        self,
        n_optimize: int = 10,
        n_offspring: int = 100,
        opt_iterations: int = 50,
        differential_weight: float = 0.5,
        cross_probability: float = 0.9,
        base_vector: str = "rand",
        init_stdev: float = 1.0,
    ):
        self.n_optimize = n_optimize
        self.n_offspring = n_offspring
        self.opt_iterations = opt_iterations
        self.F = differential_weight
        self.CR = cross_probability
        self.base_vector = base_vector
        self.init_stdev = init_stdev

    def __call__(self, forest: Forest, X, y) -> torch.Tensor:
        X = _to_device(X)
        y = _to_device(y)

        fitnesses = -forest.SR_fitness(X, y)
        fitnesses[torch.isnan(fitnesses)] = -torch.inf

        n_opt = min(self.n_optimize, forest.pop_size)
        top_indices = torch.argsort(fitnesses.cpu(), descending=True)[:n_opt].tolist()

        tree_indices: List[int] = []
        positions_list: List[np.ndarray] = []
        x0_list: List[np.ndarray] = []
        for idx in top_indices:
            positions, x0 = _extract_constants(forest, idx)
            if len(positions) > 0:
                tree_indices.append(idx)
                positions_list.append(positions)
                x0_list.append(x0)

        if not x0_list:
            return fitnesses

        bopt = BatchedDE(
            x0_list,
            self.n_offspring,
            differential_weight=self.F,
            cross_probability=self.CR,
            base_vector=self.base_vector,
            init_stdev=self.init_stdev,
            device=forest.batch_node_value.device,
        )
        builder = _BatchForestBuilder(
            forest, tree_indices, positions_list, self.n_offspring
        )

        # initial evaluation
        init_pop = bopt.ask()
        bf = builder.update_and_build(init_pop)
        mse_flat = bf.SR_fitness(X, y)
        mse_flat = torch.nan_to_num(mse_flat, nan=float("inf"), posinf=float("inf"))
        bopt.tell(init_pop, mse_flat.view(bopt.N, self.n_offspring))

        for _ in range(self.opt_iterations):
            pop_batch = bopt.ask()
            bf = builder.update_and_build(pop_batch)
            mse_flat = bf.SR_fitness(X, y)
            mse_flat = torch.nan_to_num(
                mse_flat, nan=float("inf"), posinf=float("inf")
            )
            bopt.tell(pop_batch, mse_flat.view(bopt.N, self.n_offspring))

        for i, idx in enumerate(tree_indices):
            best_x, best_mse = bopt.get_best(i)
            _write_constants(forest, idx, positions_list[i], best_x)
            fitnesses[idx] = -best_mse

        return fitnesses
