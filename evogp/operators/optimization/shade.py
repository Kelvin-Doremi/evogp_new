"""
SHADE (Success-History based Adaptive DE) constant optimization for GP trees.

Ported from evox SHADE (Tanabe & Fukunaga, CEC 2013).
Uses current-to-pbest/1 mutation with adaptive F and CR driven by a
success-history memory, batched across N optimizers.
"""
from typing import List, Tuple

import numpy as np
import torch

from .base import BaseOptimization
from .cmaes import _BatchForestBuilder, _to_device, _extract_constants, _write_constants
from evogp.core import Forest


class BatchedSHADE:
    """N SHADE instances batched into (N, ...) tensors.

    Each instance maintains its own population and a circular memory of
    successful F / CR values.  Mutation uses current-to-pbest/1.
    """

    def __init__(
        self,
        x0_list: List[np.ndarray],
        pop_size: int,
        memory_size: int = None,
        p_best_rate: float = 0.1,
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
        self.p_best_min = max(2, int(p_best_rate * pop_size))

        if memory_size is None:
            memory_size = pop_size
        self.memory_size = memory_size

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

        self.M_F = torch.full((N, memory_size), 0.5, device=device)
        self.M_CR = torch.full((N, memory_size), 0.5, device=device)
        self._mem_pos = 0

        self._F_used = None
        self._CR_used = None

        self._best_x = [np.array(x0, dtype=np.float64) for x0 in x0_list]
        self._best_mse = torch.full((N,), float("inf"), device=device)

    # ── ask / tell ──────────────────────────────────────────────

    def ask(self) -> torch.Tensor:
        if not self._initialized:
            return self.pop.clone()

        N, ps, D = self.N, self.pop_size, self.D
        device = self.device

        # adaptive F (Cauchy) and CR (Normal) from memory
        mem_idx = torch.randint(0, self.memory_size, (N, ps), device=device)
        M_F_sel = torch.gather(self.M_F, 1, mem_idx)
        M_CR_sel = torch.gather(self.M_CR, 1, mem_idx)

        u = torch.rand(N, ps, device=device)
        F = M_F_sel + 0.1 * torch.tan(torch.pi * (u - 0.5))
        F = torch.clamp(F, min=1e-6, max=1.0)

        CR = M_CR_sel + 0.1 * torch.randn(N, ps, device=device)
        CR = torch.clamp(CR, min=0.0, max=1.0)

        self._F_used = F
        self._CR_used = CR

        # current-to-pbest/1
        p = self.p_best_min
        sorted_idx = torch.argsort(self.pop_fitness, dim=1)
        pbest_rank = torch.randint(0, p, (N, ps), device=device)
        pbest_idx = torch.gather(sorted_idx, 1, pbest_rank)
        x_pbest = torch.gather(
            self.pop, 1, pbest_idx.unsqueeze(-1).expand(-1, -1, D)
        )

        r1 = torch.randint(0, ps, (N, ps), device=device)
        r2 = torch.randint(0, ps, (N, ps), device=device)
        x_r1 = torch.gather(self.pop, 1, r1.unsqueeze(-1).expand(-1, -1, D))
        x_r2 = torch.gather(self.pop, 1, r2.unsqueeze(-1).expand(-1, -1, D))

        F3 = F.unsqueeze(-1)
        mutant = self.pop + F3 * (x_pbest - self.pop) + F3 * (x_r1 - x_r2)

        cross_mask = torch.rand(N, ps, D, device=device) < CR.unsqueeze(-1)
        j_rand = torch.randint(0, D, (N, ps, 1), device=device)
        cross_mask.scatter_(2, j_rand, True)

        trial = torch.where(cross_mask, mutant, self.pop)
        return trial * self.dim_mask.unsqueeze(1)

    def tell(self, populations: torch.Tensor, mse: torch.Tensor):
        N, ps = self.N, self.pop_size

        if not self._initialized:
            self.pop_fitness = mse.clone()
            self._initialized = True
        else:
            better = mse < self.pop_fitness

            # vectorised memory update
            delta = torch.where(
                better, self.pop_fitness - mse, torch.zeros_like(mse)
            )
            delta_sum = delta.sum(dim=1, keepdim=True).clamp(min=1e-30)
            w = delta / delta_sum

            S_F = torch.where(better, self._F_used, torch.zeros_like(self._F_used))
            S_CR = torch.where(
                better, self._CR_used, torch.zeros_like(self._CR_used)
            )

            new_M_F = (w * S_F ** 2).sum(dim=1) / (w * S_F).sum(dim=1).clamp(
                min=1e-30
            )
            new_M_CR = (w * S_CR).sum(dim=1)

            has_success = better.any(dim=1)
            pos = self._mem_pos % self.memory_size
            self.M_F[:, pos] = torch.where(has_success, new_M_F, self.M_F[:, pos])
            self.M_CR[:, pos] = torch.where(
                has_success, new_M_CR, self.M_CR[:, pos]
            )
            self._mem_pos += 1

            self.pop = torch.where(better.unsqueeze(-1), populations, self.pop)
            self.pop_fitness = torch.where(better, mse, self.pop_fitness)

        # track best
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

class SHADEOptimization(BaseOptimization):
    """Optimize constants in GP trees using batched SHADE.

    Args:
        n_optimize:     How many top individuals to optimize.
        n_offspring:    SHADE population size per individual.
        opt_iterations: Number of SHADE generations.
        memory_size:    Success-history memory size (None → pop_size).
        p_best_rate:    Fraction of top individuals for pbest selection.
        init_stdev:     Std of initial population around x0.
    """

    def __init__(
        self,
        n_optimize: int = 10,
        n_offspring: int = 100,
        opt_iterations: int = 50,
        memory_size: int = None,
        p_best_rate: float = 0.1,
        init_stdev: float = 1.0,
    ):
        self.n_optimize = n_optimize
        self.n_offspring = n_offspring
        self.opt_iterations = opt_iterations
        self.memory_size = memory_size
        self.p_best_rate = p_best_rate
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

        bopt = BatchedSHADE(
            x0_list,
            self.n_offspring,
            memory_size=self.memory_size,
            p_best_rate=self.p_best_rate,
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
