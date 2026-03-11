#!/usr/bin/env python
"""
GP symbolic regression with configurable selection strategy.

Supports both ParetoTournamentSelection and standard TournamentSelection.
Always enables enable_pareto_front so the per-size archive is maintained
regardless of the selection operator, allowing fair Pareto front comparison.

Usage:
  python experiments/regressor_pareto.py --dataset 9 --seed 0 --gpu 0 --optimizer openes --selection pareto --pareto_tsize 3
  python experiments/regressor_pareto.py --dataset 9 --seed 0 --gpu 0 --optimizer openes --selection standard
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from evogp.operators import (
    DefaultCrossover,
    DefaultMutation,
    ParetoTournamentSelection,
    TournamentSelection,
)
from evogp.estimators import Regressor
from evogp.core import GenerateDescriptor
from evogp.operators.optimization import (
    CMAESOptimization,
    OpenESOptimization,
    SNESOptimization,
    XNESOptimization,
    DEOptimization,
    SHADEOptimization,
)

OPTIMIZER_CLASSES = {
    "none":   None,
    "cmaes":  CMAESOptimization,
    "openes": OpenESOptimization,
    "snes":   SNESOptimization,
    "xnes":   XNESOptimization,
    "de":     DEOptimization,
    "shade":  SHADEOptimization,
}

# ========== 常数优化超参数 ==========
OPT_N = 5000
OPT_OFFSPRING = 20
OPT_ITERS = 10
OPT_INTERVAL = 10

# ========== GP 超参数 ==========
POP_SIZE = 50000
GENERATION_LIMIT = 400
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1
INJECT_RATE = 0.05

# ── Pareto 选择参数 ──
DEFAULT_PARETO_TOURNAMENT_SIZE = 3
PARETO_ELITE_RATE = 0.25
PARETO_SURVIVOR_RATE = 0.95

# ── Standard 选择参数 ──
STANDARD_TOURNAMENT_SIZE = 20
STANDARD_ELITE_RATE = 0.10
STANDARD_SURVIVOR_RATE = 0.5

# ========== 算子配置 ==========
OPS = {"+": 0.25, "-": 0.15, "*": 0.30, "sin": 0.10, "cos": 0.10, "exp": 0.05, "abs": 0.05}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def extract_pareto_front(model, X_val, y_val):
    """Extract Pareto front from the model's per-size archive.

    Returns (sizes, train_mses, val_mses) — only non-dominated points
    in the (size, train_mse) space.
    """
    algo = model.algorithm
    if not getattr(algo, "enable_pareto_front", False):
        return np.array([]), np.array([]), np.array([])

    pf = algo.pareto_front
    fitness_arr = pf.fitness.cpu().numpy()
    valid = np.isfinite(fitness_arr)
    all_sizes = np.where(valid)[0]
    all_train_mses = -fitness_arr[all_sizes]

    pareto_idx = []
    best_mse = float("inf")
    for i, (s, m) in enumerate(zip(all_sizes, all_train_mses)):
        if m < best_mse:
            pareto_idx.append(i)
            best_mse = m

    pareto_sizes = all_sizes[pareto_idx]
    pareto_train_mses = all_train_mses[pareto_idx]

    pareto_val_mses = []
    for s in pareto_sizes:
        tree = pf.solution[int(s)]
        with torch.no_grad():
            pred = tree.forward(X_val)
            vmse = torch.mean((pred - y_val) ** 2).item()
        if not np.isfinite(vmse):
            vmse = 1e10
        pareto_val_mses.append(vmse)

    return pareto_sizes, pareto_train_mses, np.array(pareto_val_mses)


def main():
    parser = argparse.ArgumentParser(
        description="GP regression with configurable selection strategy")
    parser.add_argument("--dataset", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--optimizer", type=str, default="openes",
        choices=list(OPTIMIZER_CLASSES),
    )
    parser.add_argument(
        "--selection", type=str, default="pareto",
        choices=["pareto", "standard"],
    )
    parser.add_argument(
        "--pareto_tsize", type=int, default=DEFAULT_PARETO_TOURNAMENT_SIZE,
        help="Tournament size for ParetoTournamentSelection",
    )
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.selection == "pareto":
        config_name = f"{args.optimizer}_pareto_k{args.pareto_tsize}"
    else:
        config_name = f"{args.optimizer}_standard"
    print(f"[dataset={args.dataset}  seed={args.seed}  gpu={args.gpu}  "
          f"optimizer={args.optimizer}  selection={args.selection}  "
          f"pareto_tsize={args.pareto_tsize}]")

    # ── paths ─────────────────────────────────────────────────────
    data_dir = os.path.join(PROJECT_ROOT, "data", "datasets")
    out_dir = os.path.join(
        PROJECT_ROOT, "data", "results_pareto",
        config_name, str(args.dataset), f"seed_{args.seed}",
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────
    X = pd.read_csv(os.path.join(data_dir, f"{args.dataset}_features.csv")).to_numpy(dtype=np.float32)
    y = pd.read_csv(os.path.join(data_dir, f"{args.dataset}_targets.csv")).to_numpy(dtype=np.float32)

    n_samples = X.shape[0]
    n_train = int(n_samples * 0.8)
    indices = np.random.RandomState(42).permutation(n_samples)

    X_train = torch.FloatTensor(X[indices[:n_train]]).to(device).contiguous()
    X_val = torch.FloatTensor(X[indices[n_train:]]).to(device).contiguous()
    y_train = torch.FloatTensor(y[indices[:n_train]]).to(device).contiguous()
    y_val = torch.FloatTensor(y[indices[n_train:]]).to(device).contiguous()

    input_dim, output_dim = X.shape[1], y.shape[1]
    print(f"  features={input_dim}  outputs={output_dim}  "
          f"train={X_train.shape[0]}  val={X_val.shape[0]}")

    # ── descriptor ────────────────────────────────────────────────
    descriptor = GenerateDescriptor(
        max_tree_len=256,
        input_len=input_dim,
        output_len=1,
        using_funcs=OPS,
        max_layer_cnt=8,
        const_samples=[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3],
        layer_leaf_prob=0.3,
    )

    # ── optimizer ─────────────────────────────────────────────────
    opt_cls = OPTIMIZER_CLASSES[args.optimizer]
    optimizer = None
    if opt_cls is not None:
        optimizer = opt_cls(
            n_optimize=OPT_N,
            n_offspring=OPT_OFFSPRING,
            opt_iterations=OPT_ITERS,
        )

    # ── selection operator ────────────────────────────────────────
    if args.selection == "pareto":
        elite_rate = PARETO_ELITE_RATE
        selection = ParetoTournamentSelection(
            tournament_size=args.pareto_tsize,
            survivor_rate=PARETO_SURVIVOR_RATE,
        )
    else:
        elite_rate = STANDARD_ELITE_RATE
        selection = TournamentSelection(
            tournament_size=STANDARD_TOURNAMENT_SIZE,
            survivor_rate=STANDARD_SURVIVOR_RATE,
        )

    # ── train each output dim ─────────────────────────────────────
    all_histories = []
    all_summaries = []

    for out_idx in range(output_dim):
        ty = y_train[:, out_idx].contiguous().view(-1, 1)
        vy = y_val[:, out_idx].contiguous()
        vy_2d = vy.view(-1, 1)
        t0 = time.time()

        model = Regressor(
            descriptor,
            DefaultCrossover(crossover_rate=CROSSOVER_RATE),
            DefaultMutation(
                mutation_rate=MUTATION_RATE,
                descriptor=descriptor.update(max_layer_cnt=3),
            ),
            selection,
            pop_size=POP_SIZE,
            elite_rate=elite_rate,
            generation_limit=GENERATION_LIMIT,
            print_mse=True,
            print_mse_prefix=f"  [out{out_idx}] ",
            enable_pareto_front=True,
            inject_rate=INJECT_RATE,
            optimizer=optimizer,
            optim_interval=OPT_INTERVAL,
        )
        model.fit(X_train, ty, X_val=X_val, y_val=vy_2d)
        elapsed = time.time() - t0

        # ── extract Pareto front ──────────────────────────────────
        pf_sizes, pf_train, pf_val = extract_pareto_front(model, X_val, vy_2d)

        np.savez(
            os.path.join(out_dir, f"pareto_front_out{out_idx}.npz"),
            sizes=pf_sizes,
            train_mses=pf_train,
            val_mses=pf_val,
        )

        final_train_mse = model.history["train_mse"][-1] if model.history["train_mse"] else float("nan")
        final_val_mse = model.history["val_mse"][-1] if model.history["val_mse"] else float("nan")

        expr = ""
        try:
            expr = str(model.best_tree.to_sympy_expr())
        except Exception:
            pass

        print(f"  out{out_idx}: train_mse={final_train_mse:.6f}  "
              f"val_mse={final_val_mse:.6f}  pareto_pts={len(pf_sizes)}  "
              f"time={elapsed:.1f}s")

        all_histories.append(model.history)
        all_summaries.append({
            "output_idx": out_idx,
            "train_mse": final_train_mse,
            "val_mse": final_val_mse,
            "elapsed": elapsed,
            "n_pareto_points": int(len(pf_sizes)),
            "expression": expr,
        })

    # ── save metrics ──────────────────────────────────────────────
    np.savez(
        os.path.join(out_dir, "metrics.npz"),
        **{f"train_mse_out{i}": np.array(h["train_mse"])
           for i, h in enumerate(all_histories)},
        **{f"val_mse_out{i}": np.array(h["val_mse"])
           for i, h in enumerate(all_histories)},
        **{f"avg_tree_size_out{i}": np.array(h["avg_tree_size"])
           for i, h in enumerate(all_histories)},
    )

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "dataset": args.dataset,
            "seed": args.seed,
            "optimizer": args.optimizer,
            "selection": args.selection,
            "pareto_tsize": args.pareto_tsize,
            "config": config_name,
            "summaries": all_summaries,
        }, f, indent=2)

    print(f"  -> saved to {out_dir}")


if __name__ == "__main__":
    main()
