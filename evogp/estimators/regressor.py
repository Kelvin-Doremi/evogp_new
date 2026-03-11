import torch

from evogp.workflows import GeneticProgramming
from evogp.operators import BaseMutation, BaseCrossover, BaseSelection
from evogp.operators.optimization.base import BaseOptimization
from evogp.core import Forest, GenerateDescriptor


class Regressor:
    """GP regressor with optional per-step constant optimization.

    Each ``step`` does:
      1. (optional) constant optimization on top-N individuals
      2. GP step: selection -> crossover -> mutation
      3. Re-evaluate the new population
      4. Track best individual & per-generation history
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
        optimizer: BaseOptimization = None,
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

        self._pre_opt = optimizer
        self._opt_interval = max(1, optim_interval)
        self._step_count = 0

        self.best_tree = None
        self.best_fitness = float("-inf")

        self.history = {"train_mse": [], "val_mse": [], "avg_tree_size": []}

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

    def fit(self, X, y, X_val=None, y_val=None):
        self._step_count = 0
        self.history = {"train_mse": [], "val_mse": [], "avg_tree_size": []}

        self.fitnesses = -self.algorithm.forest.SR_fitness(X, y)
        self.fitnesses[torch.isnan(self.fitnesses)] = -torch.inf

        for gen in range(self.generation_limit):
            self.step(X, y)

            # ── record history ───────────────────────────────────
            train_mse = (
                -self.best_fitness.item()
                if hasattr(self.best_fitness, "item")
                else -self.best_fitness
            )
            avg_size = (
                self.algorithm.forest.batch_subtree_size[:, 0]
                .float().mean().item()
            )
            self.history["train_mse"].append(train_mse)
            self.history["avg_tree_size"].append(avg_size)

            if X_val is not None and y_val is not None and self.best_tree is not None:
                with torch.no_grad():
                    pred_v = self.best_tree.forward(X_val)
                    vmse = torch.mean((pred_v - y_val) ** 2).item()
                self.history["val_mse"].append(vmse)

            if self.print_mse:
                vmse_str = ""
                if self.history["val_mse"]:
                    vmse_str = f"  val_mse = {self.history['val_mse'][-1]:.6f}"
                print(
                    f"{self.print_mse_prefix}Generation {gen}: "
                    f"train_mse = {train_mse:.6f}{vmse_str}  "
                    f"avg_size = {avg_size:.1f}"
                )

            if (
                self.fitness_target is not None
                and self.best_fitness >= self.fitness_target
            ):
                print("Fitness target reached!")
                break

    def predict(self, X):
        return self.best_tree.forward(X)
