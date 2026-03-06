"""Boosted (Residual) GP Regressor.

Implements the Residual GP / Boosted GP paradigm:
    y = w1*f1(x) + w2*f2(x) + ... + wk*fk(x)

Each round fits the residual from previous rounds. After all rounds,
a linear regression re-optimizes the weights w1..wk.
"""
import torch
import numpy as np

from evogp.operators import BaseMutation, BaseCrossover, BaseSelection
from evogp.core import GenerateDescriptor, Forest
from .regressor import Regressor


class BoostedRegressor:
    """Residual GP: iteratively fit residuals, then combine with OLS.

    Args:
        n_terms:       Number of additive terms (boosting rounds).
        descriptor:    GenerateDescriptor shared by all rounds.
        crossover:     Crossover operator.
        mutation:      Mutation operator.
        selection:     Selection operator.
        pop_size:      Population size per round.
        regressor_kwargs: Extra keyword arguments forwarded to each
            :class:`Regressor` (e.g. ``elite_rate``, ``generation_limit``,
            ``optim_steps``, ``enable_pareto_front``, …).
        phase2_kwargs: If provided, a second-phase Regressor is created
            for each round with these kwargs (inheriting the phase-1
            forest). Set to ``None`` to skip phase 2.
    """

    def __init__(
        self,
        n_terms: int,
        descriptor: GenerateDescriptor,
        crossover: BaseCrossover,
        mutation: BaseMutation,
        selection: BaseSelection,
        *,
        pop_size: int = 5000,
        regressor_kwargs: dict = None,
        phase2_kwargs: dict = None,
    ):
        self.n_terms = n_terms
        self.descriptor = descriptor
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.pop_size = pop_size
        self.regressor_kwargs = regressor_kwargs or {}
        self.phase2_kwargs = phase2_kwargs

        self.terms = []         # list of Tree
        self.weights = None     # (n_terms,) or (n_terms, output_len) after OLS
        self.bias = None        # scalar or (output_len,)

    def fit(self, X, y):
        """Fit the boosted model.

        Args:
            X: (n_samples, input_dim) tensor on CUDA.
            y: (n_samples, output_dim) tensor on CUDA.
        """
        residual = y.clone()
        self.terms = []

        for t in range(self.n_terms):
            print(f"\n===== Boosting round {t + 1}/{self.n_terms} =====")
            mse_before = torch.mean(residual ** 2).item()
            print(f"  残差 MSE = {mse_before:.6f}")

            model = Regressor(
                self.descriptor,
                self.crossover,
                self.mutation,
                self.selection,
                pop_size=self.pop_size,
                print_mse_prefix=f"  [T{t+1}] ",
                **self.regressor_kwargs,
            )
            model.fit(X, residual)

            if self.phase2_kwargs is not None:
                model_p2 = Regressor(
                    self.descriptor,
                    self.phase2_kwargs.get("crossover", self.crossover),
                    self.phase2_kwargs.get("mutation", self.mutation),
                    self.phase2_kwargs.get("selection", self.selection),
                    initial_forest=model.algorithm.forest,
                    print_mse_prefix=f"  [T{t+1}-P2] ",
                    **{
                        k: v
                        for k, v in self.phase2_kwargs.items()
                        if k not in ("crossover", "mutation", "selection")
                    },
                )
                model_p2.fit(X, residual)
                if model_p2.best_fitness >= model.best_fitness:
                    model = model_p2

            best_tree = model.best_tree
            self.terms.append(best_tree)

            pred = best_tree.forward(X)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)
            residual = residual - pred

            mse_after = torch.mean(residual ** 2).item()
            print(f"  第 {t+1} 项拟合后残差 MSE = {mse_after:.6f}")
            try:
                print(f"  表达式: {best_tree.to_sympy_expr()}")
            except Exception:
                pass

        self._ols_reweight(X, y)

    def _ols_reweight(self, X, y):
        """Use OLS to re-optimize the weights: y ≈ F @ w + bias."""
        n_samples = X.shape[0]
        out_dim = y.shape[1] if y.dim() > 1 else 1
        y_flat = y.view(n_samples, out_dim)

        # (n_samples, n_terms) for each output dim
        F = torch.zeros(n_samples, self.n_terms, device=X.device)
        for i, tree in enumerate(self.terms):
            pred = tree.forward(X)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)
            F[:, i] = pred[:, 0] if pred.shape[1] == 1 else pred[:, 0]

        # OLS: [F, 1] @ [w; b] = y  =>  w, b = lstsq(A, y)
        ones = torch.ones(n_samples, 1, device=X.device)
        A = torch.cat([F, ones], dim=1)  # (n_samples, n_terms+1)

        if out_dim == 1:
            result = torch.linalg.lstsq(A, y_flat[:, 0])
            coef = result.solution
            self.weights = coef[:self.n_terms]
            self.bias = coef[self.n_terms]
        else:
            self.weights = torch.zeros(self.n_terms, out_dim, device=X.device)
            self.bias = torch.zeros(out_dim, device=X.device)
            for d in range(out_dim):
                result = torch.linalg.lstsq(A, y_flat[:, d])
                coef = result.solution
                self.weights[:, d] = coef[:self.n_terms]
                self.bias[d] = coef[self.n_terms]

        # Print results
        pred = self._predict_raw(X)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        mse = torch.mean((pred - y_flat) ** 2).item()
        print(f"\n===== OLS 重新加权后 MSE = {mse:.6f} =====")
        for i in range(self.n_terms):
            w = self.weights[i].item() if self.weights.dim() == 1 else self.weights[i]
            try:
                expr = self.terms[i].to_sympy_expr()
            except Exception:
                expr = f"f{i+1}(x)"
            print(f"  w{i+1} = {w:.6f}  *  {expr}")
        b = self.bias.item() if self.bias.dim() == 0 else self.bias
        print(f"  bias = {b}")

    def _predict_raw(self, X):
        """Predict without reshaping output."""
        n_samples = X.shape[0]
        F = torch.zeros(n_samples, self.n_terms, device=X.device)
        for i, tree in enumerate(self.terms):
            pred = tree.forward(X)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)
            F[:, i] = pred[:, 0]

        if self.weights.dim() == 1:
            return (F @ self.weights.unsqueeze(1) + self.bias).squeeze(1)
        else:
            return F @ self.weights + self.bias

    def predict(self, X):
        """Predict y given X.

        Returns:
            Tensor of shape (n_samples,) or (n_samples, output_dim).
        """
        return self._predict_raw(X)

    def get_sympy_expr(self):
        """Return the combined symbolic expression as a string."""
        parts = []
        for i in range(self.n_terms):
            w = self.weights[i].item() if self.weights.dim() == 1 else self.weights[i].tolist()
            try:
                expr = self.terms[i].to_sympy_expr()
            except Exception:
                expr = f"f{i+1}(x)"
            parts.append(f"({w}) * ({expr})")
        b = self.bias.item() if self.bias.dim() == 0 else self.bias.tolist()
        parts.append(str(b))
        return " + ".join(parts)
