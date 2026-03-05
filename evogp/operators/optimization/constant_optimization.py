"""
Constant optimization for GP trees via pluggable algorithms.

Architecture
------------
                          ConstantOptimization
                                  |
                      algorithm_factory(x0, pop_size)
                         /                  \\
                   CMAAdapter          EvoxAdapter
                   (pycma)         (evox 1.x, PyTorch)

Two adapter modes — both do cross-individual batched evaluation
---------------------------------------------------------------
CMAAdapter  (ask/tell):  Explicit ask/tell loop collects all offspring
    and evaluates in ONE SR_fitness call per iteration.

EvoxAdapter (step + threading):  Uses evox's evaluate() override pattern
    (same as StdWorkflow). Each adapter.step() runs in a thread; evaluate()
    blocks until the main thread finishes one batched SR_fitness and
    distributes the results back.

Both modes track best constants via get_best() → (best_x, best_mse).
"""
from typing import List, Tuple, Callable, Any
import threading
import numpy as np
import torch

from .base import BaseOptimization
from evogp.core import Forest, NType


# ═══════════════════════════════════════════════════════════════════════════
# Adapters
# ═══════════════════════════════════════════════════════════════════════════

class CMAAdapter:
    """Wraps pycma's CMAEvolutionStrategy into ask/tell/get_best."""

    def __init__(self, x0: np.ndarray, pop_size: int, sigma0: float = 0.5):
        import cma
        self._es = cma.CMAEvolutionStrategy(x0, sigma0, {
            "popsize": pop_size,
            "verbose": -1,
        })

    def ask(self) -> List[np.ndarray]:
        return [np.asarray(s) for s in self._es.ask()]

    def tell(self, solutions: List[np.ndarray], mse_list: List[float]) -> None:
        self._es.tell(solutions, mse_list)

    def get_best(self) -> Tuple[np.ndarray, float]:
        return np.asarray(self._es.result.xbest), float(self._es.result.fbest)


class EvoxAdapter:
    """Wraps an evox 1.x algorithm (PyTorch-based) via evaluate override.

    Same pattern as evox.workflows.StdWorkflow: subclass the algorithm,
    override evaluate(). When step() runs internally, it samples a population,
    calls evaluate(pop) — we receive the population (ask) and return
    fitness (tell) in one shot. No manual decomposition of step() needed.

    Supports different evox algorithms automatically:
      - CMAES-like (mean_init, sigma)
      - DE-like    (lb, ub, mean)
    """

    def __init__(
        self,
        x0: np.ndarray,
        pop_size: int,
        algorithm_class: Any,
        device: str = None,
        **kwargs,
    ):
        from evox.core.components import Algorithm
        import inspect

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dim = len(x0)
        dev = torch.device(device)
        x0_t = torch.tensor(x0, dtype=torch.float32, device=dev)
        params = set(inspect.signature(algorithm_class.__init__).parameters.keys())

        algo_kwargs: dict = {"device": dev}

        if "mean_init" in params:
            algo_kwargs.update(
                mean_init=x0_t,
                sigma=kwargs.pop("sigma", 0.5),
                pop_size=pop_size,
            )
        elif "lb" in params:
            lb_off = float(kwargs.pop("lb", -5.0))
            ub_off = float(kwargs.pop("ub", 5.0))
            lb_t = x0_t + lb_off
            ub_t = x0_t + ub_off
            algo_kwargs.update(pop_size=pop_size, lb=lb_t, ub=ub_t)
            if "mean" in params:
                algo_kwargs["mean"] = x0_t
                if "stdev" in params and "stdev" not in kwargs:
                    algo_kwargs["stdev"] = torch.full((dim,), 0.5, device=dev)
        else:
            algo_kwargs["pop_size"] = pop_size

        algo_kwargs.update(kwargs)
        algo = algorithm_class(**algo_kwargs)

        adapter = self

        class _Algo(type(algo)):
            def __init__(self_algo):
                super(Algorithm, self_algo).__init__()
                self_algo.__dict__.update(algo.__dict__)

            def evaluate(self_algo, pop: torch.Tensor) -> torch.Tensor:
                mse = adapter._eval_fn(pop)
                idx = torch.argmin(mse).item()
                if float(mse[idx]) < adapter._best_mse:
                    adapter._best_mse = float(mse[idx])
                    adapter._best_x = pop[idx].detach().cpu().numpy().astype(np.float64)
                return mse

        self._algo = _Algo()
        self._eval_fn = None
        self._best_x = np.array(x0, dtype=np.float64)
        self._best_mse = float("inf")

    def set_eval_fn(self, fn):
        """Set evaluation: (pop: Tensor[N, dim]) -> mse: Tensor[N]."""
        self._eval_fn = fn

    def step(self):
        """One optimisation step (sample → evaluate → update)."""
        self._algo.step()

    def get_best(self) -> Tuple[np.ndarray, float]:
        return self._best_x, self._best_mse


# ═══════════════════════════════════════════════════════════════════════════
# Factory helpers
# ═══════════════════════════════════════════════════════════════════════════

def cma_factory(sigma0: float = 0.5) -> Callable:
    """Return a factory that creates CMAAdapter instances."""
    def factory(x0: np.ndarray, pop_size: int) -> CMAAdapter:
        return CMAAdapter(x0, pop_size, sigma0=sigma0)
    return factory


def evox_factory(algorithm_class: Any, **kwargs) -> Callable:
    """Return a factory that creates EvoxAdapter instances (evox >= 1.0, PyTorch).

    Auto-detects algorithm constructor style (CMAES vs DE etc.).

    Examples::

        from evox.algorithms import CMAES, DE
        evox_factory(CMAES, sigma=0.5)
        evox_factory(DE, lb=-10, ub=10)
    """
    def factory(x0: np.ndarray, pop_size: int) -> EvoxAdapter:
        return EvoxAdapter(x0, pop_size, algorithm_class, **kwargs)
    return factory


# ═══════════════════════════════════════════════════════════════════════════
# Core optimization operator
# ═══════════════════════════════════════════════════════════════════════════

class ConstantOptimization(BaseOptimization):
    """
    Optimize constants in GP trees using a pluggable ask-tell algorithm.

    Each optimization iteration builds a Forest of size
    (n_active_individuals * n_offspring) and evaluates it in one SR_fitness
    call on GPU, then distributes the MSE results back to each algorithm.

    Args:
        n_optimize:         How many top individuals to optimize per generation.
        n_offspring:        CMA/ES population size per individual.
        opt_iterations:     How many ask-tell iterations to run.
        algorithm_factory:  Callable(x0, pop_size) -> adapter with ask/tell/get_best.
                            Defaults to pycma CMA-ES (sigma0=0.5).
    """

    def __init__(
        self,
        n_optimize: int,
        n_offspring: int = 10,
        opt_iterations: int = 20,
        algorithm_factory: Callable = None,
    ):
        self.n_optimize = n_optimize
        self.n_offspring = n_offspring
        self.opt_iterations = opt_iterations
        self.algorithm_factory = algorithm_factory or cma_factory()

    def __call__(self, forest: Forest, X, y) -> torch.Tensor:
        X = _check_tensor(X)
        y = _check_tensor(y)

        # ── 1. Evaluate the whole population ──
        fitnesses = -forest.SR_fitness(X, y)          # fitness = -MSE
        fitnesses[torch.isnan(fitnesses)] = -torch.inf

        # ── 2. Pick top-n individuals that have constants ──
        n_opt = min(self.n_optimize, forest.pop_size)
        top_indices = torch.argsort(fitnesses.cpu(), descending=True)[:n_opt].tolist()

        opt_states: List[Tuple[int, np.ndarray, Any]] = []  # (tree_idx, const_positions, algo)
        for idx in top_indices:
            positions, x0 = _extract_constants(forest, idx)
            if len(positions) == 0:
                continue
            algo = self.algorithm_factory(x0, self.n_offspring)
            opt_states.append((idx, positions, algo))

        if not opt_states:
            return fitnesses

        # ── 3. Optimization loop ──
        if hasattr(opt_states[0][2], 'ask'):
            # ask/tell adapters (CMAAdapter): cross-individual batched evaluation
            for _ in range(self.opt_iterations):
                all_solutions = [algo.ask() for (_, _, algo) in opt_states]
                batch_forest = _build_batch_forest(forest, opt_states, all_solutions)
                mse_all = batch_forest.SR_fitness(X, y).cpu().numpy()
                mse_all = np.nan_to_num(mse_all, nan=np.inf, posinf=np.inf)
                offset = 0
                for i, (_, _, algo) in enumerate(opt_states):
                    n = len(all_solutions[i])
                    algo.tell(all_solutions[i], mse_all[offset:offset + n].tolist())
                    offset += n
        else:
            # step-based adapters (EvoxAdapter): cross-individual batched via threading
            # Each adapter.step() runs in a thread; evaluate() blocks until the main
            # thread finishes one batched SR_fitness call and distributes results back.
            n_adapters = len(opt_states)
            pop_slots = [None] * n_adapters
            fit_slots = [None] * n_adapters
            step_errors = [None] * n_adapters
            lock = threading.Lock()
            ready_cnt = [0]
            all_asked = threading.Event()
            told = [threading.Event() for _ in range(n_adapters)]

            def _coordinated_eval(idx):
                def fn(pop):
                    pop_slots[idx] = pop.detach()
                    with lock:
                        ready_cnt[0] += 1
                        if ready_cnt[0] == n_adapters:
                            all_asked.set()
                    told[idx].wait()
                    told[idx].clear()
                    return fit_slots[idx]
                return fn

            def _safe_step(idx, adapter):
                try:
                    adapter.step()
                except Exception as e:
                    step_errors[idx] = e
                    with lock:
                        ready_cnt[0] += 1
                        if ready_cnt[0] == n_adapters:
                            all_asked.set()

            for i, (_, _, adapter) in enumerate(opt_states):
                adapter.set_eval_fn(_coordinated_eval(i))

            for _ in range(self.opt_iterations):
                all_asked.clear()
                ready_cnt[0] = 0

                threads = [
                    threading.Thread(target=_safe_step, args=(i, a), daemon=True)
                    for i, (_, _, a) in enumerate(opt_states)
                ]
                for t in threads:
                    t.start()

                all_asked.wait()

                first_err = next((e for e in step_errors if e is not None), None)
                if first_err is not None:
                    for ev in told:
                        ev.set()
                    for t in threads:
                        t.join()
                    raise first_err

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                batch = _build_batch_forest_torch(forest, opt_states, pop_slots)
                mse_all = batch.SR_fitness(X, y)
                mse_all = torch.nan_to_num(mse_all, nan=float("inf"), posinf=float("inf"))

                offset = 0
                for i in range(n_adapters):
                    sz = pop_slots[i].shape[0]
                    fit_slots[i] = mse_all[offset:offset + sz]
                    offset += sz

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                for ev in told:
                    ev.set()
                for t in threads:
                    t.join()

        # ── 4. Write best constants back ──
        for tree_idx, positions, algo in opt_states:
            best_x, best_mse = algo.get_best()
            _write_constants(forest, tree_idx, positions, best_x)
            fitnesses[tree_idx] = -best_mse  # fitness = -MSE

        return fitnesses


# For backwards compatibility
CMAESOptimization = ConstantOptimization


# ═══════════════════════════════════════════════════════════════════════════
# Helpers (module-private)
# ═══════════════════════════════════════════════════════════════════════════

def _check_tensor(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=torch.float32, device="cuda", requires_grad=False)
    return x.to("cuda").detach().requires_grad_(False)


def _extract_constants(
    forest: Forest, tree_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (positions, values) of constant nodes in tree_idx."""
    node_type = forest.batch_node_type[tree_idx].cpu().numpy()
    node_value = forest.batch_node_value[tree_idx].cpu().numpy()
    tree_size = int(forest.batch_subtree_size[tree_idx, 0].item())
    mask = np.array([(node_type[i] & 0x7F) == NType.CONST for i in range(tree_size)])
    positions = np.where(mask)[0]
    values = node_value[positions].astype(np.float64).copy()
    return positions, values


def _build_batch_forest(
    forest: Forest,
    opt_states: List[Tuple[int, np.ndarray, Any]],
    all_solutions: List[List[np.ndarray]],
) -> Forest:
    """Assemble one Forest containing every offspring of every individual."""
    n_trees = sum(len(sols) for sols in all_solutions)
    device = forest.batch_node_value.device
    mtl = forest.max_tree_len

    batch_value = torch.zeros((n_trees, mtl), dtype=torch.float32, device=device)
    batch_type = torch.zeros((n_trees, mtl), dtype=torch.int16, device=device)
    batch_size = torch.zeros((n_trees, mtl), dtype=torch.int16, device=device)

    offset = 0
    for i, (tree_idx, positions, _) in enumerate(opt_states):
        base_val = forest.batch_node_value[tree_idx]        # (mtl,)
        base_typ = forest.batch_node_type[tree_idx]          # (mtl,)
        base_siz = forest.batch_subtree_size[tree_idx]       # (mtl,)

        for sol in all_solutions[i]:
            batch_value[offset] = base_val
            batch_type[offset] = base_typ
            batch_size[offset] = base_siz
            for k, pos in enumerate(positions):
                batch_value[offset, pos] = float(sol[k])
            offset += 1

    return Forest(forest.input_len, forest.output_len, batch_value, batch_type, batch_size)


def _build_batch_forest_torch(
    forest: Forest,
    opt_states: List[Tuple[int, np.ndarray, Any]],
    pop_slots: List[torch.Tensor],
) -> Forest:
    """Assemble one Forest from torch tensor populations (EvoxAdapter path)."""
    n_trees = sum(pop.shape[0] for pop in pop_slots)
    device = forest.batch_node_value.device
    mtl = forest.max_tree_len

    batch_value = torch.zeros((n_trees, mtl), dtype=torch.float32, device=device)
    batch_type = torch.zeros((n_trees, mtl), dtype=torch.int16, device=device)
    batch_size = torch.zeros((n_trees, mtl), dtype=torch.int16, device=device)

    offset = 0
    for i, (tree_idx, positions, _) in enumerate(opt_states):
        pop = pop_slots[i]
        sz = pop.shape[0]
        batch_value[offset:offset + sz] = forest.batch_node_value[tree_idx]
        batch_type[offset:offset + sz] = forest.batch_node_type[tree_idx]
        batch_size[offset:offset + sz] = forest.batch_subtree_size[tree_idx]
        for k, pos in enumerate(positions):
            batch_value[offset:offset + sz, pos] = pop[:, k].to(device=device, dtype=torch.float32)
        offset += sz

    return Forest(forest.input_len, forest.output_len, batch_value, batch_type, batch_size)


def _write_constants(
    forest: Forest, tree_idx: int, positions: np.ndarray, constants: np.ndarray
) -> None:
    for i, pos in enumerate(positions):
        forest.batch_node_value[tree_idx, pos] = float(constants[i])
