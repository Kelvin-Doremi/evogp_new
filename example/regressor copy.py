import torch

torch.random.manual_seed(2)
torch.cuda.manual_seed(2)

from evogp.pipeline import StandardPipeline
from evogp.tree import Forest, GenerateDescriptor
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
    CombinedMutation,
    DeleteMutation,
    TournamentSelection,
)
from evogp.problem import SymbolicRegression


def func(x):
    val = x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)
    return val.reshape(-1)


problem = SymbolicRegression(
    func=func, num_inputs=2, num_data=100, lower_bounds=-5, upper_bounds=5
)

descriptor = GenerateDescriptor(
    max_tree_len=32,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=5,
    const_samples=[-1, 0, 1],
    layer_leaf_prob=0.3,
)


algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(pop_size=1000, descriptor=descriptor),
    crossover=DefaultCrossover(),
    mutation=DefaultMutation(
        mutation_rate=0.1, descriptor=descriptor.update(max_layer_cnt=3)
    ),
    selection=TournamentSelection(
        tournament_size=20, survivor_rate=0.5, elite_rate=0.1
    ),
    enable_pareto_front=False,
)

pipeline = StandardPipeline(
    algorithm,
    problem,
    generation_limit=300,
)

best = pipeline.run()

sympy_expression = best.to_sympy_expr()
print(sympy_expression)

# print(algorithm.pareto_front)