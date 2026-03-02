import torch

from ..algorithm import GeneticProgramming, BaseMutation, BaseCrossover, BaseSelection
from evogp.tree import Forest


class Regressor:
    def __init__(
        self,
        initial_forest: Forest,
        crossover: BaseCrossover,
        mutation: BaseMutation,
        selection: BaseSelection,
        fitness_target: float = None,
        generation_limit: int = 100,
        optimize_constants: bool = False,
        bfgs_top_k: int = 10,
        bfgs_max_iter: int = 100,
        bfgs_backend: str = "auto",
        bfgs_async: bool = False,
        verbose: bool = False,
        bfgs_start_gen: int = 0,
        optimize_method: str = "bfgs",
    ):
        self.verbose = verbose
        self.bfgs_start_gen = bfgs_start_gen
        self.optimize_method = optimize_method
        self.algorithm = GeneticProgramming(
            initial_forest, crossover, mutation, selection
        )
        self.fitness_target = fitness_target
        self.generation_limit = generation_limit
        self.optimize_constants = optimize_constants
        self.bfgs_top_k = bfgs_top_k
        self.bfgs_max_iter = bfgs_max_iter
        self.bfgs_backend = bfgs_backend
        self.bfgs_async = bfgs_async

        self.best_tree = None
        self.best_fitness = float("-inf")
        self.fitness = None
        self._optimization_pool = None

    def step(self, X, y, generation_cnt: int = 0):
        # 1. 注入上一轮异步优化完成的结果
        if self._optimization_pool is not None:
            self._inject_optimized_trees(X, y)

        # 2. 评估适应度
        fitnesses = -self.algorithm.forest.SR_fitness(X, y)
        fitnesses[torch.isnan(fitnesses)] = -torch.inf

        # 3. 常数优化（max_iter<=0 时跳过，generation_cnt < bfgs_start_gen 时不优化）
        if self.optimize_constants and self.bfgs_max_iter > 0 and generation_cnt >= self.bfgs_start_gen:
            labels = y.view(-1, 1) if y.dim() == 1 else y
            top_k = min(self.bfgs_top_k, len(self.algorithm.forest))
            _, top_indices = torch.topk(fitnesses.cpu(), top_k)

            if self.bfgs_async:
                for idx in top_indices.tolist():
                    self._optimization_pool.submit(
                        self.algorithm.forest[idx], float(fitnesses[idx].item())
                    )
            else:
                from ..optim import optimize_tree_constants
                for idx in top_indices.tolist():
                    try:
                        opt_tree, _ = optimize_tree_constants(
                            self.algorithm.forest[idx], X, labels,
                            max_iter=self.bfgs_max_iter, backend=self.bfgs_backend,
                            method=self.optimize_method,
                        )
                        self.algorithm.forest[idx] = opt_tree
                    except (NotImplementedError, ValueError):
                        pass
                fitnesses = -self.algorithm.forest.SR_fitness(X, y)
                fitnesses[torch.isnan(fitnesses)] = -torch.inf

        # 4. 更新最优 & 遗传操作
        cpu_fitness = fitnesses.cpu()
        best_idx, best_fitness = int(torch.argmax(cpu_fitness)), torch.max(cpu_fitness)
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_tree = self.algorithm.forest[best_idx]

        self.fitness = cpu_fitness  # 供 fit() 中 verbose 打印
        self.algorithm.step(fitnesses)

    def _inject_optimized_trees(self, X, y):
        results = self._optimization_pool.get_optimized()
        if not results:
            return
        forest = self.algorithm.forest
        fitnesses = -forest.SR_fitness(X, y)
        fitnesses[torch.isnan(fitnesses)] = -torch.inf
        k = min(len(results), len(forest))
        _, worst_indices = torch.topk(fitnesses.cpu(), k, largest=False)
        for (opt_tree, opt_fitness), idx in zip(results, worst_indices.tolist()):
            if opt_fitness <= float(fitnesses[idx].item()):
                continue
            with torch.no_grad():
                pred = opt_tree.forward(X)
                if pred.numel() > 1 and pred.std().item() < 1e-6:
                    continue
            forest[idx] = opt_tree

    def fit(self, X, y):
        # 异步模式：启动优化池（若 bfgs_start_gen > 0 也提前启动，到代数后再提交）
        if self.bfgs_async and self.optimize_constants and self.bfgs_max_iter > 0 and self.generation_limit > self.bfgs_start_gen:
            from ..optim import OptimizationPool
            labels = y.view(-1, 1) if y.dim() == 1 else y
            self._optimization_pool = OptimizationPool(
                X, labels,
                max_iter=self.bfgs_max_iter,
                backend=self.bfgs_backend,
                max_queue_size=self.bfgs_top_k * 3,
                method=self.optimize_method,
            )
            self._optimization_pool.start()

        try:
            generation_cnt = 0
            while True:
                self.step(X, y, generation_cnt)
                if self.verbose and self.fitness is not None:
                    best_mse = -float(torch.max(self.fitness).item())
                    print(f"  Gen {generation_cnt}: MSE={best_mse:.6f}")

                if (
                    self.fitness_target is not None
                    and self.best_fitness >= self.fitness_target
                ):
                    print("Fitness target reached!")
                    break

                generation_cnt += 1
                if generation_cnt >= self.generation_limit:
                    print("Generation limit reached!")
                    break
        finally:
            if self._optimization_pool is not None:
                self._optimization_pool.stop()
                self._optimization_pool = None

    def predict(self, X):
        return self.best_tree.forward(X)
