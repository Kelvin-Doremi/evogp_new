from evogp.core import Forest


class BaseOptimization:
    """Base class for constant optimization operators."""

    def __call__(self, forest: Forest, X, y):
        """
        Optimize constants in the forest and return fitnesses.

        Modifies forest in place and returns the fitness tensor.

        Args:
            forest: The population of trees to optimize.
            X: Input tensor for fitness evaluation.
            y: Target tensor for fitness evaluation.

        Returns:
            Fitness tensor of shape (pop_size,).
        """
        raise NotImplementedError
