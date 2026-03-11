import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

from evogp.workflows import StandardWorkflow, GeneticProgramming
from evogp.core import Forest, GenerateDescriptor
from evogp.operators import (
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
    CombinedMutation,
    DeleteMutation,
)
from evogp.problems import Classification



dataset_name = ["iris", "wine", "breast_cancer", "digits"]


multi_output = True

problem = Classification(multi_output, dataset="iris")

descriptor = GenerateDescriptor(
    max_tree_len=64,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=4,
    const_samples=[-1, 0, 1],
)


algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(pop_size=1000, descriptor=descriptor),
    crossover=DefaultCrossover(),
        mutation=CombinedMutation(
        [
            DefaultMutation(
                mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
            ),
            DeleteMutation(mutation_rate=0.8),
        ]
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

best.to_png("./imgs/classifier_tree.png")
sympy_expression = best.to_sympy_expr()
print(sympy_expression)

print(algorithm.pareto_front)
