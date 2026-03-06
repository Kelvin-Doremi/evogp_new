"""
CMA-ES constant optimization for GP trees.

Uses a batched CMA-ES implementation: all optimized individuals run as one
set of tensor operations (pad to max_dim), with a single batched SR_fitness
call per iteration for evaluation.
"""
import math
from typing import List, Tuple

import numpy as np
import torch

from .base import BaseOptimization
from evogp.core import Forest, NType


# ═══════════════════════════════════════════════════════════════════
# Batched CMA-ES (all optimizers in one set of tensor ops)
# ═══════════════════════════════════════════════════════════════════

class BatchedCMAES:
    """N CMA-ES instances batched into (N, ...) tensors.

    Different optimizers may have different dimensionalities; all are padded
    to ``max_dim`` so that every operation is a single batched kernel call.
    """

    def __init__(self, x0_list: List[np.ndarray], pop_size: int,
                 sigma: float = 1.0, device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        N = len(x0_list)
        dims = [len(x0) for x0 in x0_list]
        D = max(dims)
        self.N, self.D, self.pop_size = N, D, pop_size
        self.mu = pop_size // 2
        self.dims = torch.tensor(dims, device=device)

        # (N, D) mask: 1 for valid positions
        self.dim_mask = torch.zeros(N, D, device=device)
        for i, d in enumerate(dims):
            self.dim_mask[i, :d] = 1.0
        # (N, D, D) mask for covariance matrix
        self.C_mask = self.dim_mask.unsqueeze(-1) * self.dim_mask.unsqueeze(1)
        self.C_eye_pad = torch.eye(D, device=device).unsqueeze(0) * (1 - self.C_mask)

        # mean: (N, 1, D)
        mean = torch.zeros(N, D, device=device)
        for i, x0 in enumerate(x0_list):
            mean[i, :len(x0)] = torch.tensor(x0, dtype=torch.float32)
        self.mean = mean.unsqueeze(1)

        self.sigma = torch.full((N,), sigma, device=device)
        self.iteration = 0

        # weights: shared (1, 1, mu)
        w = (torch.log(torch.tensor((pop_size + 1) / 2.0, device=device))
             - torch.log(torch.arange(1, self.mu + 1, device=device,
                                      dtype=torch.float32)))
        w = w / w.sum()
        self.weights = w.view(1, 1, self.mu)
        mu_eff = float(w.sum() ** 2 / (w ** 2).sum())

        # per-optimizer hyperparams: (N,)
        df = self.dims.float()
        self.c_sigma = (mu_eff + 2) / (df + mu_eff + 5)
        self.d_sigma = (1 + 2 * torch.clamp(
            torch.sqrt((mu_eff - 1) / (df + 1)) - 1, min=0) + self.c_sigma)
        self.c_c = (mu_eff + 2) / (df + 4 + 2 * mu_eff / df)
        self.c_1 = 2 / ((df + 1.3) ** 2 + mu_eff)
        self.c_mu = torch.minimum(
            1 - self.c_1,
            2 * (mu_eff - 2 + 1 / mu_eff) / ((df + 2) ** 2 + mu_eff))
        self.chi_n = torch.sqrt(df) * (1 - 1 / (4 * df) + 1 / (21 * df ** 2))
        self._sqrt_cs = torch.sqrt(torch.abs(
            self.c_sigma * (2 - self.c_sigma) * mu_eff))
        self._sqrt_cc = torch.sqrt(torch.abs(
            self.c_c * (2 - self.c_c) * mu_eff))

        # state
        eye = torch.eye(D, device=device).unsqueeze(0)
        self.C = eye.expand(N, -1, -1).clone()
        self.C_invsqrt = eye.expand(N, -1, -1).clone()
        self.BD = eye.expand(N, -1, -1).clone()
        self.p_sigma = torch.zeros(N, D, device=device)
        self.p_c = torch.zeros(N, D, device=device)

        self._best_x = [np.array(x0, dtype=np.float64) for x0 in x0_list]
        self._best_mse = torch.full((N,), float("inf"), device=device)

    def ask(self) -> torch.Tensor:
        """Sample populations. Returns (N, pop_size, D)."""
        z = torch.randn(self.N, self.pop_size, self.D, device=self.device)
        pop = (self.mean
               + self.sigma.view(-1, 1, 1) * torch.matmul(z, self.BD))
        return pop * self.dim_mask.unsqueeze(1)

    def tell(self, populations: torch.Tensor, mse: torch.Tensor):
        """Update state. populations: (N, pop_size, D), mse: (N, pop_size)."""
        self.iteration += 1
        it = self.iteration
        N, D, mu = self.N, self.D, self.mu

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

        # sort ascending → select top-mu
        order = torch.argsort(mse, dim=1)
        pop_sorted = torch.gather(
            populations, 1, order.unsqueeze(-1).expand(-1, -1, D))
        selected = pop_sorted[:, :mu]

        # mean update
        new_mean = (torch.bmm(self.weights.expand(N, -1, -1),
                               selected - self.mean) + self.mean)
        delta = (new_mean - self.mean).squeeze(1)

        # p_sigma
        Cinv_d = torch.bmm(self.C_invsqrt,
                            delta.unsqueeze(-1)).squeeze(-1)
        self.p_sigma = ((1 - self.c_sigma).unsqueeze(-1) * self.p_sigma
                        + self._sqrt_cs.unsqueeze(-1) * Cinv_d
                          / self.sigma.unsqueeze(-1))
        self.p_sigma *= self.dim_mask

        # h_sigma
        norm_ps = torch.norm(self.p_sigma, dim=-1)
        denom = 1 - (1 - self.c_sigma) ** (2 * it)
        h_sigma = (norm_ps / torch.sqrt(torch.clamp(denom, min=1e-30))
                   < (1.4 + 2 / (self.dims.float() + 1)) * self.chi_n
                   ).float()

        # p_c
        self.p_c = ((1 - self.c_c).unsqueeze(-1) * self.p_c
                     + h_sigma.unsqueeze(-1) * self._sqrt_cc.unsqueeze(-1)
                       * delta / self.sigma.unsqueeze(-1))
        self.p_c *= self.dim_mask

        # covariance update
        yw = (selected - self.mean) / self.sigma.view(-1, 1, 1)
        rank_one = torch.bmm(self.p_c.unsqueeze(-1), self.p_c.unsqueeze(1))
        rank_mu = torch.bmm(
            yw.transpose(1, 2) * self.weights.view(1, 1, mu), yw)

        c1 = self.c_1.view(-1, 1, 1)
        cm = self.c_mu.view(-1, 1, 1)
        cc = self.c_c.view(-1, 1, 1)
        hs = h_sigma.view(-1, 1, 1)

        self.C = ((1 - c1 - cm) * self.C
                  + c1 * (rank_one + (1 - hs) * cc * (2 - cc) * self.C)
                  + cm * rank_mu)
        self.C = self.C * self.C_mask + self.C_eye_pad

        # sigma
        self.sigma *= torch.exp(
            self.c_sigma / self.d_sigma * (norm_ps / self.chi_n - 1))

        self.mean = new_mean

        # eigendecomposition (batched)
        C_sym = (self.C + self.C.transpose(1, 2)) / 2
        eigval, eigvec = torch.linalg.eigh(C_sym)
        eigval = torch.clamp(eigval, min=1e-8)
        inv_sqrt = 1.0 / torch.sqrt(eigval)
        self.C_invsqrt = torch.bmm(
            eigvec * inv_sqrt.unsqueeze(1), eigvec.transpose(1, 2))
        self.BD = eigvec * torch.sqrt(eigval).unsqueeze(1)

    def get_best(self, i: int) -> Tuple[np.ndarray, float]:
        return self._best_x[i], float(self._best_mse[i])


# ═══════════════════════════════════════════════════════════════════
# Batch Forest builder (pre-allocated, reused across iterations)
# ═══════════════════════════════════════════════════════════════════

class _BatchForestBuilder:
    """Pre-allocate one Forest worth of tensors; only rewrite constant
    columns each iteration."""

    def __init__(self, forest: Forest, tree_indices, positions_list, pop_size):
        n_trees = len(tree_indices) * pop_size
        dev = forest.batch_node_value.device
        mtl = forest.max_tree_len

        self.batch_value = torch.zeros(
            (n_trees, mtl), dtype=torch.float32, device=dev)
        self.batch_type = torch.zeros(
            (n_trees, mtl), dtype=torch.int16, device=dev)
        self.batch_size = torch.zeros(
            (n_trees, mtl), dtype=torch.int16, device=dev)

        self._slices = []
        self._pos_tensors = []
        self._base_values = []
        offset = 0
        for idx, positions in zip(tree_indices, positions_list):
            self.batch_value[offset:offset + pop_size] = \
                forest.batch_node_value[idx]
            self.batch_type[offset:offset + pop_size] = \
                forest.batch_node_type[idx]
            self.batch_size[offset:offset + pop_size] = \
                forest.batch_subtree_size[idx]
            pos_t = torch.tensor(positions, dtype=torch.long, device=dev)
            self._slices.append((offset, pop_size))
            self._pos_tensors.append(pos_t)
            self._base_values.append(
                forest.batch_node_value[idx].clone())
            offset += pop_size

        self._meta = (forest.input_len, forest.output_len)

    def update_and_build(self, pop_batch: torch.Tensor) -> Forest:
        """pop_batch: (N, pop_size, D) or list of (pop_size, d_i)."""
        for i in range(len(self._slices)):
            off, sz = self._slices[i]
            pos_t = self._pos_tensors[i]
            self.batch_value[off:off + sz] = self._base_values[i]
            if isinstance(pop_batch, torch.Tensor):
                self.batch_value[off:off + sz, pos_t] = \
                    pop_batch[i, :, :len(pos_t)].float()
            else:
                self.batch_value[off:off + sz, pos_t] = \
                    pop_batch[i][:, :len(pos_t)].float()

        return Forest(self._meta[0], self._meta[1],
                      self.batch_value, self.batch_type, self.batch_size)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _to_device(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=torch.float32, device="cuda",
                            requires_grad=False)
    return x.to("cuda").detach().requires_grad_(False)


def _extract_constants(
    forest: Forest, tree_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (positions, values) of constant nodes in *tree_idx*."""
    node_type = forest.batch_node_type[tree_idx].cpu().numpy()
    node_value = forest.batch_node_value[tree_idx].cpu().numpy()
    tree_size = int(forest.batch_subtree_size[tree_idx, 0].item())
    mask = (node_type[:tree_size] & 0x7F) == NType.CONST
    positions = np.where(mask)[0]
    values = node_value[positions].astype(np.float64).copy()
    return positions, values


def _write_constants(
    forest: Forest, tree_idx: int, positions: np.ndarray,
    constants: np.ndarray
) -> None:
    dev = forest.batch_node_value.device
    pos_t = torch.tensor(positions, dtype=torch.long, device=dev)
    val_t = torch.tensor(constants, dtype=torch.float32, device=dev)
    forest.batch_node_value[tree_idx, pos_t] = val_t


# ═══════════════════════════════════════════════════════════════════
# Public operator
# ═══════════════════════════════════════════════════════════════════

class CMAESOptimization(BaseOptimization):
    """Optimize constants in GP trees using batched CMA-ES.

    Selects the top *n_optimize* individuals that contain constants,
    creates a batched CMA-ES over all of them, and runs *opt_iterations*
    ask/eval/tell loops.  Evaluation is done with a single ``SR_fitness``
    call per iteration.

    Args:
        n_optimize:     How many top individuals to optimize.
        n_offspring:    CMA-ES population size per individual (λ).
        opt_iterations: Number of CMA-ES iterations.
        sigma:          Initial step size.
    """

    def __init__(
        self,
        n_optimize: int = 10,
        n_offspring: int = 100,
        opt_iterations: int = 50,
        sigma: float = 1.0,
    ):
        self.n_optimize = n_optimize
        self.n_offspring = n_offspring
        self.opt_iterations = opt_iterations
        self.sigma = sigma

    def __call__(self, forest: Forest, X, y) -> torch.Tensor:
        X = _to_device(X)
        y = _to_device(y)

        # 1. evaluate whole population
        fitnesses = -forest.SR_fitness(X, y)
        fitnesses[torch.isnan(fitnesses)] = -torch.inf

        # 2. pick top-N individuals that have constants
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

        # 3. create batched CMA-ES + forest builder
        bcma = BatchedCMAES(
            x0_list, self.n_offspring, sigma=self.sigma,
            device=forest.batch_node_value.device)
        builder = _BatchForestBuilder(
            forest, tree_indices, positions_list, self.n_offspring)

        # 4. optimization loop
        for _ in range(self.opt_iterations):
            pop_batch = bcma.ask()
            bf = builder.update_and_build(pop_batch)
            mse_flat = bf.SR_fitness(X, y)
            mse_flat = torch.nan_to_num(
                mse_flat, nan=float("inf"), posinf=float("inf"))
            mse_batch = mse_flat.view(bcma.N, self.n_offspring)
            bcma.tell(pop_batch, mse_batch)

        # 5. write best constants back
        for i, idx in enumerate(tree_indices):
            best_x, best_mse = bcma.get_best(i)
            _write_constants(forest, idx, positions_list[i], best_x)
            fitnesses[idx] = -best_mse

        return fitnesses


ConstantOptimization = CMAESOptimization
