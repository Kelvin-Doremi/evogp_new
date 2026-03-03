from problem import MujocoProblem

import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

from evogp.core import Forest, GenerateDescriptor
from evogp.workflows import GeneticProgramming
from evogp.operators import (
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
    DeleteMutation,
    CombinedMutation,
)
from evogp.workflows import StandardWorkflow

problem = MujocoProblem(
    "SwimmerSwimmer6",
    max_episode_length=1000,
)

descriptor = GenerateDescriptor(
    max_tree_len=256,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=6,
    const_range=[-1, 1],
    sample_cnt=100,
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
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
)

pipeline = StandardWorkflow(
    algorithm,
    problem,
    generation_limit=100,
)

pipeline.run()
