"""
Separable NES constant optimization for GP trees.

Ported from evox SeparableNES (jmlr.org/papers/volume15/wierstra14a).
Uses a batched implementation: all optimized individuals run as one
set of tensor operations (pad to max_dim), with a single batched SR_fitness
call per iteration for evaluation.

Separable NES maintains per-dimension step sizes (diagonal covariance),
making it O(D) per update — much cheaper than CMA-ES's O(D^2) covariance.
"""
import math
from typing import List, Tuple

import numpy as np
import torch

from .base import BaseOptimization
from .cmaes import _BatchForestBuilder, _to_device, _extract_constants, _write_constants
from evogp.core import Forest


# ═══════════════════════════════════════════════════════════════════
# Batched Separable NES (all optimizers in one set of tensor ops)
# ═══════════════════════════════════════════════════════════════════

class BatchedSeparableNES:
    """N Separable NES instances batched into (N, ...) tensors.

    Different optimizers may have different dimensionalities; all are padded
    to ``max_dim`` so that every operation is a single batched kernel call.

    Uses rank-based natural gradient with per-dimension step sizes (sigma).
    """

    def __init__(self, x0_list: List[np.ndarray], pop_size: int,
                 sigma: float = 0.1, learning_rate_mean: float = 1.0,
                 learning_rate_var: float = None,
                 device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        N = len(x0_list)
        dims = [len(x0) for x0 in x0_list]
        D = max(dims) if dims else 1
        self.N, self.D, self.pop_size = N, D, pop_size
        self.dims = torch.tensor(dims, device=device)

        # (N, D) mask: 1 for valid positions
        self.dim_mask = torch.zeros(N, D, device=device)
        for i, d in enumerate(dims):
            self.dim_mask[i, :d] = 1.0

        # mean: (N, 1, D)
        mean = torch.zeros(N, D, device=device)
        for i, x0 in enumerate(x0_list):
            mean[i, :len(x0)] = torch.tensor(x0, dtype=torch.float32)
        self.mean = mean.unsqueeze(1)

        # sigma: (N, D) per-dimension standard deviation
        self.sigma = torch.full((N, D), sigma, device=device) * self.dim_mask

        # learning rates
        self.lr_mean = learning_rate_mean
        if learning_rate_var is None:
            df = self.dims.float().clamp(min=1)
            self.lr_var = ((3 + torch.log(df)) / 5 / torch.sqrt(df)).view(-1, 1)
        else:
            self.lr_var = torch.full((N, 1), learning_rate_var, device=device)

        # recombination weights: (pop_size,) shared across all optimizers
        ranks = torch.arange(1, pop_size + 1, device=device, dtype=torch.float32)
        w = torch.clamp(math.log(pop_size / 2 + 1) - torch.log(ranks), min=0)
        w = w / w.sum() - 1.0 / pop_size
        self.weights = w

        self._noise = None
        self._best_x = [np.array(x0, dtype=np.float64) for x0 in x0_list]
        self._best_mse = torch.full((N,), float("inf"), device=device)

    def ask(self) -> torch.Tensor:
        """Sample populations. Returns (N, pop_size, D)."""
        noise = torch.randn(self.N, self.pop_size, self.D, device=self.device)
        noise = noise * self.dim_mask.unsqueeze(1)
        self._noise = noise
        pop = self.mean + noise * self.sigma.unsqueeze(1)
        return pop * self.dim_mask.unsqueeze(1)

    def tell(self, populations: torch.Tensor, mse: torch.Tensor):
        """Update state. populations: (N, pop_size, D), mse: (N, pop_size).

        ``mse`` should be positive (lower = better), consistent with SR_fitness.
        """
        N, D = self.N, self.D

        # track best
        best_idx = torch.argmin(mse, dim=1)
        best_mse = mse[torch.arange(N, device=self.device), best_idx]
        improved = best_mse < self._best_mse
        if improved.any():
            self._best_mse = torch.where(improved, best_mse, self._best_mse)
            for i in torch.where(improved)[0].tolist():
                d = int(self.dims[i])
                self._best_x[i] = (populations[i, best_idx[i], :d]
                                    .cpu().numpy().astype(np.float64))

        # sort ascending (lowest MSE = best first) per optimizer
        order = torch.argsort(mse, dim=1)
        sorted_noise = torch.gather(
            self._noise, 1, order.unsqueeze(-1).expand(-1, -1, D))

        # weighted natural gradients: (N, D)
        w = self.weights.view(1, -1, 1)  # (1, pop_size, 1)
        grad_mean = (w * sorted_noise).sum(dim=1)
        grad_sigma = (w * (sorted_noise * sorted_noise - 1)).sum(dim=1)

        grad_mean = grad_mean * self.dim_mask
        grad_sigma = grad_sigma * self.dim_mask

        # update mean: (N, 1, D)
        self.mean = (self.mean
                     + self.lr_mean * self.sigma.unsqueeze(1)
                       * grad_mean.unsqueeze(1))

        # update sigma: (N, D) with clamped exponent for stability
        exponent = torch.clamp(self.lr_var / 2 * grad_sigma, -1.0, 1.0)
        self.sigma = torch.clamp(
            self.sigma * torch.exp(exponent), 1e-8, 1e8) * self.dim_mask

    def get_best(self, i: int) -> Tuple[np.ndarray, float]:
        return self._best_x[i], float(self._best_mse[i])


# ═══════════════════════════════════════════════════════════════════
# Public operator
# ═══════════════════════════════════════════════════════════════════

class SeparableNESOptimization(BaseOptimization):
    """Optimize constants in GP trees using batched Separable NES.

    Much faster than CMA-ES: O(D) per update instead of O(D^2).
    Uses rank-based natural gradient with adaptive per-dimension step sizes.

    Args:
        n_optimize:             How many top individuals to optimize.
        n_offspring:            Population size per individual (lambda).
        opt_iterations:         Number of NES iterations.
        sigma:                  Initial per-dimension step size.
        learning_rate_mean:     Learning rate for mean update.
        learning_rate_var:      Learning rate for sigma update (None = auto).
    """

    def __init__(
        self,
        n_optimize: int = 10,
        n_offspring: int = 100,
        opt_iterations: int = 50,
        sigma: float = 0.1,
        learning_rate_mean: float = 1.0,
        learning_rate_var: float = None,
    ):
        self.n_optimize = n_optimize
        self.n_offspring = n_offspring
        self.opt_iterations = opt_iterations
        self.sigma = sigma
        self.lr_mean = learning_rate_mean
        self.lr_var = learning_rate_var

    def __call__(self, forest: Forest, X, y) -> torch.Tensor:
        X = _to_device(X)
        y = _to_device(y)

        fitnesses = -forest.SR_fitness(X, y)
        fitnesses[torch.isnan(fitnesses)] = -torch.inf

        n_opt = min(self.n_optimize, forest.pop_size)
        top_indices = torch.argsort(
            fitnesses.cpu(), descending=True)[:n_opt].tolist()

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

        bopt = BatchedSeparableNES(
            x0_list, self.n_offspring,
            sigma=self.sigma,
            learning_rate_mean=self.lr_mean,
            learning_rate_var=self.lr_var,
            device=forest.batch_node_value.device)
        builder = _BatchForestBuilder(
            forest, tree_indices, positions_list, self.n_offspring)

        for _ in range(self.opt_iterations):
            pop_batch = bopt.ask()
            bf = builder.update_and_build(pop_batch)
            mse_flat = bf.SR_fitness(X, y)
            mse_flat = torch.nan_to_num(
                mse_flat, nan=float("inf"), posinf=float("inf"))
            mse_batch = mse_flat.view(bopt.N, self.n_offspring)
            bopt.tell(pop_batch, mse_batch)

        for i, idx in enumerate(tree_indices):
            best_x, best_mse = bopt.get_best(i)
            _write_constants(forest, idx, positions_list[i], best_x)
            fitnesses[idx] = -best_mse

        return fitnesses
