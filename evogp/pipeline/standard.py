import time
import numpy as np
import torch

from ..algorithm import GeneticProgramming
from ..problem import BaseProblem
from . import BasePipeline


class StandardPipeline(BasePipeline):
    def __init__(
        self,
        algorithm: GeneticProgramming,
        problem: BaseProblem,
        fitness_target: float = None,
        generation_limit: int = 100,
        time_limit: int = None,
        is_show_details: bool = True,
        valid_fitness_boundry: float = 1e8,
        optimize_constants: bool = False,
        bfgs_top_k: int = 10,
        bfgs_max_iter: int = 100,
        bfgs_backend: str = "auto",
        bfgs_async: bool = False,
        bfgs_start_gen: int = 0,
        optimize_method: str = "bfgs",
    ):
        """
        Args:
            optimize_constants: 是否启用常数优化
            bfgs_async: True=异步优化池（GP 不阻塞），False=同步优化（每代等待）
            bfgs_start_gen: 从该代数开始执行常数优化，之前不优化
            optimize_method: "bfgs" 或 "es"，es 通常更快
        """

        self.algorithm = algorithm
        self.bfgs_start_gen = bfgs_start_gen
        self.optimize_method = optimize_method
        self.problem = problem
        self.fitness_target = fitness_target
        self.generation_limit = generation_limit
        self.time_limit = time_limit
        self.is_show_details = is_show_details
        self.valid_fitness_boundry = valid_fitness_boundry
        self.optimize_constants = optimize_constants
        self.bfgs_top_k = bfgs_top_k
        self.bfgs_max_iter = bfgs_max_iter
        self.bfgs_backend = bfgs_backend
        self.bfgs_async = bfgs_async

        self.best_tree = None
        self.best_fitness = float("-inf")
        self.fitness = None
        self.generation_timestamp = None
        self._optimization_pool = None
        self.generation_cnt = 0

    def step(self):
        # 1. 注入上一轮异步优化完成的结果（替换最差个体）
        if self._optimization_pool is not None:
            self._inject_optimized_trees()

        # 2. 评估适应度
        fitnesses = self.problem.evaluate(self.algorithm.forest)
        fitnesses[torch.isnan(fitnesses)] = -torch.inf

        # 3. 常数优化（max_iter<=0 时跳过，generation_cnt < bfgs_start_gen 时不优化）
        if self.optimize_constants and self.bfgs_max_iter > 0 and self.generation_cnt >= self.bfgs_start_gen and hasattr(self.problem, "datapoints") and hasattr(self.problem, "labels"):
            inputs = self.problem.datapoints
            labels = self.problem.labels
            top_k = min(self.bfgs_top_k, len(self.algorithm.forest))
            _, top_indices = torch.topk(fitnesses.cpu(), top_k)

            if self.bfgs_async:
                # 异步：提交到优化池，不等待
                for idx in top_indices.tolist():
                    self._optimization_pool.submit(self.algorithm.forest[idx], float(fitnesses[idx].item()))
            else:
                # 同步：原地优化并写回
                from ..optim import optimize_tree_constants
                for idx in top_indices.tolist():
                    try:
                        opt_tree, _ = optimize_tree_constants(
                            self.algorithm.forest[idx], inputs, labels,
                            max_iter=self.bfgs_max_iter, backend=self.bfgs_backend,
                            method=self.optimize_method,
                        )
                        self.algorithm.forest[idx] = opt_tree
                    except (NotImplementedError, ValueError):
                        pass
                fitnesses = self.problem.evaluate(self.algorithm.forest)
                fitnesses[torch.isnan(fitnesses)] = -torch.inf

        # 4. 更新最优 & 遗传操作
        cpu_fitness = fitnesses.cpu()
        best_idx, best_fitness = int(torch.argmax(cpu_fitness)), torch.max(cpu_fitness)
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_tree = self.algorithm.forest[best_idx]

        self.algorithm.step(fitnesses)
        return cpu_fitness

    def _inject_optimized_trees(self):
        """将异步优化完成的个体注入种群，仅当优化后优于当前最差且非常数解时才替换"""
        results = self._optimization_pool.get_optimized()
        if not results:
            return

        forest = self.algorithm.forest
        fitnesses = self.problem.evaluate(forest)
        fitnesses[torch.isnan(fitnesses)] = -torch.inf

        k = min(len(results), len(forest))
        _, worst_indices = torch.topk(fitnesses.cpu(), k, largest=False)
        datapoints = self.problem.datapoints
        for (opt_tree, opt_fitness), idx in zip(results, worst_indices.tolist()):
            if opt_fitness <= float(fitnesses[idx].item()):
                continue
            with torch.no_grad():
                pred = opt_tree.forward(datapoints)
                if pred.numel() > 1 and pred.std().item() < 1e-6:
                    continue  # 常数解，不注入
            forest[idx] = opt_tree

    def run(self):
        tic = time.time()

        # 异步模式：启动优化池（max_iter<=0 或 bfgs_start_gen>=limit 时不启动）
        if self.bfgs_async and self.optimize_constants and self.bfgs_max_iter > 0 and self.generation_limit > self.bfgs_start_gen and hasattr(self.problem, "datapoints") and hasattr(self.problem, "labels"):
            from ..optim import OptimizationPool
            self._optimization_pool = OptimizationPool(
                self.problem.datapoints,
                self.problem.labels,
                max_iter=self.bfgs_max_iter,
                backend=self.bfgs_backend,
                max_queue_size=self.bfgs_top_k * 3,
                method=self.optimize_method,
            )
            self._optimization_pool.start()

        try:
            generation_cnt = 0
            while True:
                self.generation_cnt = generation_cnt
                if self.is_show_details:
                    start_time = time.time()

                self.fitness = self.step()

                if self.is_show_details:
                    self.show_details(start_time, generation_cnt, self.fitness)

                if (
                    self.fitness_target is not None
                    and self.best_fitness >= self.fitness_target
                ):
                    print("Fitness target reached!")
                    break

                if self.time_limit is not None and time.time() - tic > self.time_limit:
                    print("Time limit reached!")
                    break

                generation_cnt += 1
                if generation_cnt >= self.generation_limit:
                    print("Generation limit reached!")
                    break

            return self.best_tree
        finally:
            if self._optimization_pool is not None:
                self._optimization_pool.stop()
                self._optimization_pool = None

    def show_details(self, start_time, generation_cnt, fitnesses):

        valid_fitness = fitnesses[
            (fitnesses < self.valid_fitness_boundry)
            & (fitnesses > -self.valid_fitness_boundry)
        ]

        max_f, min_f, mean_f, std_f = (
            max(valid_fitness),
            min(valid_fitness),
            torch.mean(valid_fitness),
            torch.std(valid_fitness),
        )
        cost_time = time.time() - start_time

        print(
            f"Generation: {generation_cnt}, Cost time: {cost_time * 1000:.2f}ms\n",
            f"\tfitness: valid cnt: {len(valid_fitness)}, max: {max_f:.4f}, min: {min_f:.4f}, mean: {mean_f:.4f}, std: {std_f:.4f}\n",
        )
