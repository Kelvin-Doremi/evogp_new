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
        print_mse: bool = False,
        print_mse_prefix: str = "",
    ):
        self.algorithm = GeneticProgramming(
            initial_forest, crossover, mutation, selection
        )
        self.fitness_target = fitness_target
        self.generation_limit = generation_limit
        self.print_mse = print_mse
        self.print_mse_prefix = print_mse_prefix

        self.best_tree = None
        self.best_fitness = float("-inf")

    def step(self, X, y):
        # evaluate fitness
        fitnesses = -self.algorithm.forest.SR_fitness(X, y)
        fitnesses[torch.isnan(fitnesses)] = -torch.inf

        # update the algorithm status
        self.algorithm.step(fitnesses)

        # update the best tree info
        cpu_fitness = fitnesses.cpu()
        best_idx, best_fitness = int(torch.argmax(cpu_fitness)), torch.max(cpu_fitness)
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_tree = self.algorithm.forest[best_idx]

    def fit(self, X, y):
        generation_cnt = 0
        while True:
            self.step(X, y)

            if self.print_mse:
                mse = (
                    -self.best_fitness.item()
                    if hasattr(self.best_fitness, "item")
                    else -self.best_fitness
                )
                print(f"{self.print_mse_prefix}Generation {generation_cnt}: MSE = {mse:.6f}")

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
