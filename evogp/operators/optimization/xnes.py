"""
xNES (Exponential Natural Evolution Strategies) constant optimization for GP trees.

Ported from evox XNES (doi.org/10.1145/1830483.1830557).
Uses a batched implementation: all optimized individuals run as one
set of tensor operations (pad to max_dim), with a single batched SR_fitness
call per iteration for evaluation.

xNES maintains a full transformation matrix B (rotation + scaling),
updated via matrix exponential of the natural gradient.  More expressive
than Separable NES but O(D^2) per update.
"""
import math
from typing import List, Tuple

import numpy as np
import torch

from .base import BaseOptimization
from .cmaes import _BatchForestBuilder, _to_device, _extract_constants, _write_constants
from evogp.core import Forest


class BatchedXNES:
    """N xNES instances batched into (N, ...) tensors.

    Each instance maintains a full (D, D) transformation matrix ``B`` and a
    scalar step-size ``sigma``, updated via natural gradient with matrix
    exponential.
    """

    def __init__(
        self,
        x0_list: List[np.ndarray],
        pop_size: int,
        sigma: float = 1.0,
        learning_rate_mean: float = None,
        learning_rate_var: float = None,
        learning_rate_B: float = None,
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

        self.dim_mask = torch.zeros(N, D, device=device)
        for i, d in enumerate(dims):
            self.dim_mask[i, :d] = 1.0

        self.C_mask = self.dim_mask.unsqueeze(-1) * self.dim_mask.unsqueeze(-2)
        self.I_valid = torch.eye(D, device=device).unsqueeze(0) * self.C_mask
        self.I_pad = torch.eye(D, device=device).unsqueeze(0) * (1 - self.C_mask)

        mean = torch.zeros(N, D, device=device)
        for i, x0 in enumerate(x0_list):
            mean[i, : len(x0)] = torch.tensor(x0, dtype=torch.float32)
        self.mean = mean

        self.sigma = torch.full((N,), sigma, device=device)
        self.B = torch.eye(D, device=device).unsqueeze(0).expand(N, -1, -1).clone()

        # learning rates (per-optimizer, depend on dim)
        df = self.dims.float().clamp(min=1)
        self.lr_mean = learning_rate_mean if learning_rate_mean is not None else 1.0
        if learning_rate_var is None:
            self.lr_var = (9 + 3 * torch.log(df)) / 5 / df.pow(1.5)
        else:
            self.lr_var = torch.full((N,), learning_rate_var, device=device)
        self.lr_B = self.lr_var.clone() if learning_rate_B is None else torch.full(
            (N,), learning_rate_B, device=device
        )

        ranks = torch.arange(1, pop_size + 1, device=device, dtype=torch.float32)
        w = torch.clamp(math.log(pop_size / 2 + 1) - torch.log(ranks), min=0)
        w = w / w.sum() - 1.0 / pop_size
        self.weights = w

        self._noise = None
        self._best_x = [np.array(x0, dtype=np.float64) for x0 in x0_list]
        self._best_mse = torch.full((N,), float("inf"), device=device)

    # ── ask / tell ──────────────────────────────────────────────

    def ask(self) -> torch.Tensor:
        """Sample populations. Returns (N, pop_size, D)."""
        noise = torch.randn(self.N, self.pop_size, self.D, device=self.device)
        noise = noise * self.dim_mask.unsqueeze(1)
        self._noise = noise
        # pop = mean + sigma * (noise @ B^T)
        scaled = torch.bmm(noise, self.B.transpose(1, 2))
        pop = self.mean.unsqueeze(1) + self.sigma.view(-1, 1, 1) * scaled
        return pop * self.dim_mask.unsqueeze(1)

    def tell(self, populations: torch.Tensor, mse: torch.Tensor):
        N, D = self.N, self.D

        # track best
        best_idx = torch.argmin(mse, dim=1)
        best_mse = mse[torch.arange(N, device=self.device), best_idx]
        improved = best_mse < self._best_mse
        if improved.any():
            self._best_mse = torch.where(improved, best_mse, self._best_mse)
            for i in torch.where(improved)[0].tolist():
                d = int(self.dims[i])
                self._best_x[i] = (
                    populations[i, best_idx[i], :d].cpu().numpy().astype(np.float64)
                )

        # sort ascending (lowest MSE first)
        order = torch.argsort(mse, dim=1)
        sorted_noise = torch.gather(
            self._noise, 1, order.unsqueeze(-1).expand(-1, -1, D)
        )

        w = self.weights  # (pop_size,)

        # grad_delta = Σ w_i · z_i  →  (N, D)
        grad_delta = (w.view(1, -1, 1) * sorted_noise).sum(dim=1) * self.dim_mask

        # grad_M = (w·z^T)·z − Σw · I_valid  →  (N, D, D)
        zT = sorted_noise.transpose(1, 2)  # (N, D, pop_size)
        raw = torch.bmm(zT * w.view(1, 1, -1), sorted_noise)
        grad_M = raw - w.sum() * self.I_valid

        # grad_sigma = tr(grad_M) / dim_i  →  (N,)
        diag_M = torch.diagonal(grad_M, dim1=-2, dim2=-1)
        grad_sigma = (diag_M * self.dim_mask).sum(dim=-1) / self.dims.float().clamp(min=1)

        # grad_B = grad_M − grad_sigma · I_valid  →  (N, D, D)
        grad_B = (grad_M - grad_sigma.view(-1, 1, 1) * self.I_valid) * self.C_mask

        # ── updates ──────────────────────────────────────────────
        Bg = torch.bmm(self.B, grad_delta.unsqueeze(-1)).squeeze(-1)
        self.mean = (
            self.mean + self.lr_mean * self.sigma.unsqueeze(-1) * Bg
        ) * self.dim_mask

        self.sigma = self.sigma * torch.exp(self.lr_var / 2 * grad_sigma)

        exp_arg = self.lr_B.view(-1, 1, 1) / 2 * grad_B
        self.B = torch.bmm(self.B, torch.linalg.matrix_exp(exp_arg))
        self.B = self.B * self.C_mask + self.I_pad

    def get_best(self, i: int) -> Tuple[np.ndarray, float]:
        return self._best_x[i], float(self._best_mse[i])


# ═══════════════════════════════════════════════════════════════════
# Public operator
# ═══════════════════════════════════════════════════════════════════

class XNESOptimization(BaseOptimization):
    """Optimize constants in GP trees using batched xNES.

    xNES maintains a full transformation matrix (rotation + scale) so it
    can adapt to correlated search landscapes.  More expressive than
    Separable NES / OpenES but O(D^2) per update.

    Args:
        n_optimize:         How many top individuals to optimize.
        n_offspring:        Population size per individual (lambda).
        opt_iterations:     Number of xNES iterations.
        sigma:              Initial global step size.
        learning_rate_mean: Learning rate for mean (default 1).
        learning_rate_var:  Learning rate for sigma (None → auto).
        learning_rate_B:    Learning rate for B matrix (None → same as var).
    """

    def __init__(
        self,
        n_optimize: int = 10,
        n_offspring: int = 100,
        opt_iterations: int = 50,
        sigma: float = 1.0,
        learning_rate_mean: float = None,
        learning_rate_var: float = None,
        learning_rate_B: float = None,
    ):
        self.n_optimize = n_optimize
        self.n_offspring = n_offspring
        self.opt_iterations = opt_iterations
        self.sigma = sigma
        self.lr_mean = learning_rate_mean
        self.lr_var = learning_rate_var
        self.lr_B = learning_rate_B

    def __call__(self, forest: Forest, X, y) -> torch.Tensor:
        X = _to_device(X)
        y = _to_device(y)

        fitnesses = -forest.SR_fitness(X, y)
        fitnesses[torch.isnan(fitnesses)] = -torch.inf

        n_opt = min(self.n_optimize, forest.pop_size)
        top_indices = torch.argsort(
            fitnesses.cpu(), descending=True
        )[:n_opt].tolist()

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

        bopt = BatchedXNES(
            x0_list,
            self.n_offspring,
            sigma=self.sigma,
            learning_rate_mean=self.lr_mean,
            learning_rate_var=self.lr_var,
            learning_rate_B=self.lr_B,
            device=forest.batch_node_value.device,
        )
        builder = _BatchForestBuilder(
            forest, tree_indices, positions_list, self.n_offspring
        )

        for _ in range(self.opt_iterations):
            pop_batch = bopt.ask()
            bf = builder.update_and_build(pop_batch)
            mse_flat = bf.SR_fitness(X, y)
            mse_flat = torch.nan_to_num(
                mse_flat, nan=float("inf"), posinf=float("inf")
            )
            mse_batch = mse_flat.view(bopt.N, self.n_offspring)
            bopt.tell(pop_batch, mse_batch)

        for i, idx in enumerate(tree_indices):
            best_x, best_mse = bopt.get_best(i)
            _write_constants(forest, idx, positions_list[i], best_x)
            fitnesses[idx] = -best_mse

        return fitnesses
