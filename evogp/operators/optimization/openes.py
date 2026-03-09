"""
OpenES constant optimization for GP trees.

Ported from evox OpenES (arxiv.org/abs/1703.03864).
Uses a batched implementation: all optimized individuals run as one
set of tensor operations (pad to max_dim), with a single batched SR_fitness
call per iteration for evaluation.
"""
import math
from typing import List, Tuple

import numpy as np
import torch

from .base import BaseOptimization
from .cmaes import _BatchForestBuilder, _to_device, _extract_constants, _write_constants
from evogp.core import Forest


# ═══════════════════════════════════════════════════════════════════
# Batched OpenES (all optimizers in one set of tensor ops)
# ═══════════════════════════════════════════════════════════════════

class BatchedOpenES:
    """N OpenES instances batched into (N, ...) tensors.

    Different optimizers may have different dimensionalities; all are padded
    to ``max_dim`` so that every operation is a single batched kernel call.

    Supports optional Adam optimizer and mirrored sampling (antithetic noise).
    """

    def __init__(self, x0_list: List[np.ndarray], pop_size: int,
                 learning_rate: float = 0.05, noise_stdev: float = 0.1,
                 use_adam: bool = True, mirrored_sampling: bool = True,
                 device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        N = len(x0_list)
        dims = [len(x0) for x0 in x0_list]
        D = max(dims) if dims else 1
        self.N, self.D, self.pop_size = N, D, pop_size
        self.dims = torch.tensor(dims, device=device)
        self.mirrored_sampling = mirrored_sampling
        self.use_adam = use_adam
        self.noise_stdev = noise_stdev
        self.learning_rate = learning_rate

        if mirrored_sampling:
            assert pop_size % 2 == 0, "mirrored_sampling requires even pop_size"

        # (N, D) mask: 1 for valid positions
        self.dim_mask = torch.zeros(N, D, device=device)
        for i, d in enumerate(dims):
            self.dim_mask[i, :d] = 1.0

        # center: (N, 1, D)
        center = torch.zeros(N, D, device=device)
        for i, x0 in enumerate(x0_list):
            center[i, :len(x0)] = torch.tensor(x0, dtype=torch.float32)
        self.center = center.unsqueeze(1)

        # Adam state: (N, D)
        if use_adam:
            self.exp_avg = torch.zeros(N, D, device=device)
            self.exp_avg_sq = torch.zeros(N, D, device=device)
            self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

        self._noise = None
        self._best_x = [np.array(x0, dtype=np.float64) for x0 in x0_list]
        self._best_mse = torch.full((N,), float("inf"), device=device)

    def ask(self) -> torch.Tensor:
        """Sample populations. Returns (N, pop_size, D)."""
        if self.mirrored_sampling:
            half = self.pop_size // 2
            noise = torch.randn(self.N, half, self.D, device=self.device)
            noise = torch.cat([noise, -noise], dim=1)
        else:
            noise = torch.randn(self.N, self.pop_size, self.D, device=self.device)
        noise = noise * self.dim_mask.unsqueeze(1)
        self._noise = noise
        pop = self.center + self.noise_stdev * noise
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

        # gradient estimate: (N, D) = bmm((N, D, pop), (N, pop, 1))
        noise = self._noise
        grad = torch.bmm(
            noise.transpose(1, 2),
            mse.unsqueeze(-1),
        ).squeeze(-1)
        grad = grad / self.pop_size / self.noise_stdev
        grad = grad * self.dim_mask

        # center update
        if self.use_adam:
            self.exp_avg = self.beta1 * self.exp_avg + (1 - self.beta1) * grad
            self.exp_avg_sq = (self.beta2 * self.exp_avg_sq
                               + (1 - self.beta2) * grad * grad)
            update = self.learning_rate * self.exp_avg / (self.exp_avg_sq.sqrt() + self.eps)
        else:
            update = self.learning_rate * grad

        self.center = self.center - (update * self.dim_mask).unsqueeze(1)

    def get_best(self, i: int) -> Tuple[np.ndarray, float]:
        return self._best_x[i], float(self._best_mse[i])


# ═══════════════════════════════════════════════════════════════════
# Public operator
# ═══════════════════════════════════════════════════════════════════

class OpenESOptimization(BaseOptimization):
    """Optimize constants in GP trees using batched OpenES.

    Much faster than CMA-ES per iteration: no covariance matrix or
    eigendecomposition.  Uses simple gradient estimation via Gaussian
    noise perturbations, with optional Adam optimizer.

    Args:
        n_optimize:         How many top individuals to optimize.
        n_offspring:        Population size per individual (lambda).
        opt_iterations:     Number of OpenES iterations.
        learning_rate:      Step size (or Adam lr).
        noise_stdev:        Std of Gaussian perturbations.
        use_adam:           Use Adam optimizer (default True).
        mirrored_sampling:  Use antithetic noise pairs (default True).
    """

    def __init__(
        self,
        n_optimize: int = 10,
        n_offspring: int = 100,
        opt_iterations: int = 50,
        learning_rate: float = 0.05,
        noise_stdev: float = 0.1,
        use_adam: bool = True,
        mirrored_sampling: bool = True,
    ):
        self.n_optimize = n_optimize
        self.n_offspring = n_offspring + (n_offspring % 2) if mirrored_sampling else n_offspring
        self.opt_iterations = opt_iterations
        self.learning_rate = learning_rate
        self.noise_stdev = noise_stdev
        self.use_adam = use_adam
        self.mirrored_sampling = mirrored_sampling

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

        bopt = BatchedOpenES(
            x0_list, self.n_offspring,
            learning_rate=self.learning_rate,
            noise_stdev=self.noise_stdev,
            use_adam=self.use_adam,
            mirrored_sampling=self.mirrored_sampling,
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
