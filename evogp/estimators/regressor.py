import torch

from evogp.workflows import GeneticProgramming
from evogp.operators import BaseMutation, BaseCrossover, BaseSelection
from evogp.operators.optimization import CMAESOptimization
from evogp.core import Forest, GenerateDescriptor


class Regressor:
    """GP regressor with optional per-step constant optimization.

    Each ``step`` does:
      1. (optional) CMA-ES constant optimization on top-N individuals
      2. GP step: selection → crossover → mutation
      3. Re-evaluate the new population
      4. Track best individual

    Two phases with different intensities can be achieved by creating
    two Regressor instances and passing the first one's forest to the
    second via ``initial_forest``.
    """

    def __init__(
        self,
        descriptor: GenerateDescriptor,
        crossover: BaseCrossover,
        mutation: BaseMutation,
        selection: BaseSelection,
        *,
        initial_forest: Forest = None,
        pop_size: int = None,
        elite_rate: float = 0.0,
        fitness_target: float = None,
        generation_limit: int = 100,
        print_mse: bool = False,
        print_mse_prefix: str = "",
        enable_pareto_front: bool = False,
        inject_rate: float = 0.0,
        optim_steps: int = 0,
        optim_n: int = 500,
        optim_offspring: int = 20,
        optim_interval: int = 1,
    ):
        self.descriptor = descriptor
        self.algorithm = GeneticProgramming(
            initial_forest=initial_forest,
            crossover=crossover,
            mutation=mutation,
            selection=selection,
            descriptor=descriptor,
            pop_size=pop_size,
            elite_rate=elite_rate,
            enable_pareto_front=enable_pareto_front,
            inject_rate=inject_rate,
        )
        self.fitness_target = fitness_target
        self.generation_limit = generation_limit
        self.print_mse = print_mse
        self.print_mse_prefix = print_mse_prefix

        self._pre_opt = None
        if optim_steps > 0:
            self._pre_opt = CMAESOptimization(
                n_optimize=optim_n,
                n_offspring=optim_offspring,
                opt_iterations=optim_steps,
                sigma=1.0,
            )

        self._opt_interval = max(1, optim_interval)
        self._step_count = 0

        self.best_tree = None
        self.best_fitness = float("-inf")

    def step(self, X, y):
        if (
            self._pre_opt is not None
            and self._step_count > 0
            and self._step_count % self._opt_interval == 0
        ):
            self.fitnesses = self._pre_opt(self.algorithm.forest, X, y)
        self._step_count += 1

        self.algorithm.step(self.fitnesses)

        self.fitnesses = -self.algorithm.forest.SR_fitness(X, y)
        self.fitnesses[torch.isnan(self.fitnesses)] = -torch.inf

        cpu_fitness = self.fitnesses.cpu()
        best_idx = int(torch.argmax(cpu_fitness))
        best_fitness = torch.max(cpu_fitness)
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_tree = self.algorithm.forest[best_idx]

    def fit(self, X, y):
        self._step_count = 0
        self.fitnesses = -self.algorithm.forest.SR_fitness(X, y)
        self.fitnesses[torch.isnan(self.fitnesses)] = -torch.inf

        for gen in range(self.generation_limit):
            self.step(X, y)

            if self.print_mse:
                mse = (
                    -self.best_fitness.item()
                    if hasattr(self.best_fitness, "item")
                    else -self.best_fitness
                )
                print(f"{self.print_mse_prefix}Generation {gen}: MSE = {mse:.6f}")

            if (
                self.fitness_target is not None
                and self.best_fitness >= self.fitness_target
            ):
                print("Fitness target reached!")
                break

    def predict(self, X):
        return self.best_tree.forward(X)
