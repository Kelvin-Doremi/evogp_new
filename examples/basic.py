import torch
from evogp.core import Forest, GenerateDescriptor
from evogp.workflows import GeneticProgramming
from evogp.operators import (
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
)
from evogp.problems import SymbolicRegression
from evogp.workflows import StandardWorkflow

XOR_INPUTS = torch.tensor(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=torch.float,
    device="cuda",
)

XOR_OUTPUTS = torch.tensor(
    [[0], [1], [1], [0], [1], [0], [0], [1]],
    dtype=torch.float,
    device="cuda",
)

problem = SymbolicRegression(datapoints=XOR_INPUTS, labels=XOR_OUTPUTS)

# create decriptor for generating new trees
descriptor = GenerateDescriptor(
    max_tree_len=64,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=5,
    const_samples=[-1, 0, 1],
)

# create the algorithm
algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(pop_size=50, descriptor=descriptor),
    crossover=DefaultCrossover(),
    mutation=DefaultMutation(
        mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
    ),
    selection=DefaultSelection(),
    enable_pareto_front=True,
    elite_rate=0.01,
)

pipeline = StandardWorkflow(
    algorithm,
    problem,
    generation_limit=100,
)

best = pipeline.run()

pred_res = best.forward(XOR_INPUTS)
print(pred_res)

sympy_expression = best.to_sympy_expr()
print(sympy_expression)

# best.to_png("./imgs/xor_tree.png")
print(algorithm.pareto_front)
