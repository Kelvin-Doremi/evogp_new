"""
Regressor with Evolution Strategy (ES) constant optimization.

Extends Regressor by applying constant optimization (e.g. CMA-ES) after each
genetic step. The optimization replaces the separate evaluation phase.
"""
import torch

from evogp.workflows import GeneticProgramming
from evogp.operators import BaseMutation, BaseCrossover, BaseSelection
from evogp.operators.optimization import BaseOptimization
from evogp.core import Forest


class RegressorES:
    """
    Regressor with constant optimization via Evolution Strategy.

    After each genetic step (selection, crossover, mutation), applies constant
    optimization to the top n_optimize individuals. The optimization both
    updates constants and computes fitnesses, so no separate evaluation is needed.
    """

    def __init__(
        self,
        initial_forest: Forest,
        crossover: BaseCrossover,
        mutation: BaseMutation,
        selection: BaseSelection,
        optimization: BaseOptimization,
        elite_rate: float = 0.0,
        fitness_target: float = None,
        generation_limit: int = 100,
        print_mse: bool = False,
        print_mse_prefix: str = "",
    ):
        self.algorithm = GeneticProgramming(
            initial_forest, crossover, mutation, selection, elite_rate=elite_rate
        )
        self.optimization = optimization
        self.fitness_target = fitness_target
        self.generation_limit = generation_limit
        self.print_mse = print_mse
        self.print_mse_prefix = print_mse_prefix

        self.best_tree = None
        self.best_fitness = float("-inf")

    def step(self, X, y):
        self.algorithm.step(self.fitnesses)

        self.fitnesses = self.optimization(
            self.algorithm.forest, X, y
        )
        self.fitnesses[torch.isnan(self.fitnesses)] = -torch.inf

        cpu_fitness = self.fitnesses.cpu()
        best_idx = int(torch.argmax(cpu_fitness))
        best_fitness = torch.max(cpu_fitness)
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_tree = self.algorithm.forest[best_idx]

    def fit(self, X, y):
        generation_cnt = 0
        self.fitnesses = -self.algorithm.forest.SR_fitness(X, y)
        self.fitnesses[torch.isnan(self.fitnesses)] = -torch.inf

        while True:
            self.step(X, y)

            if self.print_mse:
                mse = (
                    -self.best_fitness.item()
                    if hasattr(self.best_fitness, "item")
                    else -self.best_fitness
                )
                print(
                    f"{self.print_mse_prefix}Generation {generation_cnt}: "
                    f"MSE = {mse:.6f}"
                )

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

    def predict(self, X):
        return self.best_tree.forward(X)
