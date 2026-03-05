import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

from evogp.workflows import StandardWorkflow
from evogp.core import Forest, GenerateDescriptor
from evogp.workflows import GeneticProgramming
from evogp.operators import (
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
    CombinedMutation,
    DeleteMutation,
)
from evogp.problems import SymbolicRegression


def func(x):
    val = x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)
    return val.reshape(-1)


problem = SymbolicRegression(
    func=func, num_inputs=2, num_data=100, lower_bounds=-5, upper_bounds=5
)

descriptor = GenerateDescriptor(
    max_tree_len=64,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=6,
    const_samples=[-1, 0, 1],
    layer_leaf_prob=0,
)


algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(pop_size=1000, descriptor=descriptor),
    crossover=DefaultCrossover(crossover_rate=0.9),
    mutation=CombinedMutation(
        [
            DefaultMutation(
                mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
            ),
            DeleteMutation(mutation_rate=0.8),
        ]
    ),
    selection=DefaultSelection(),
    enable_pareto_front=False,
    elite_rate=0.1,
)

pipeline = StandardWorkflow(
    algorithm,
    problem,
    generation_limit=150,
)

best = pipeline.run()

sympy_expression = best.to_sympy_expr()
print(sympy_expression)

# print(algorithm.pareto_front)