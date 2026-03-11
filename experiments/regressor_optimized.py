#!/usr/bin/env python
"""
GP symbolic regression experiment with configurable constant optimizer.

Usage:
  python experiments/regressor_optimized.py --dataset 9 --seed 0 --gpu 0 --optimizer openes
  python experiments/regressor_optimized.py --dataset 9 --seed 0 --gpu 0 --optimizer none
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from evogp.operators import DefaultCrossover, DefaultMutation, TournamentSelection
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

# ========== 可选优化器 ==========
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
OPT_N = 5000            # 每次优化 top-N 个个体
OPT_OFFSPRING = 20      # 每个个体的候选解数量
OPT_ITERS = 10          # 每次优化的迭代次数
OPT_INTERVAL = 10       # 每隔几代做一次优化

# ========== GP 超参数 ==========
POP_SIZE = 50000
ELITE_RATE = 0.15
GENERATION_LIMIT = 200
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 20
SURVIVOR_RATE = 0.5
INJECT_RATE = 0.05

# ========== 算子配置 ==========
OPS = {"+": 0.25, "-": 0.15, "*": 0.30, "sin": 0.10, "cos": 0.10, "exp": 0.05, "abs": 0.05}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="GP symbolic regression experiment")
    parser.add_argument("--dataset", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="openes",
        choices=list(OPTIMIZER_CLASSES),
    )
    args = parser.parse_args()

    # ── device & seed ────────────────────────────────────────────
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"[dataset={args.dataset}  seed={args.seed}  gpu={args.gpu}  "
          f"optimizer={args.optimizer}]")

    # ── paths ────────────────────────────────────────────────────
    data_dir = os.path.join(PROJECT_ROOT, "data", "datasets")
    out_dir = os.path.join(
        PROJECT_ROOT, "data", "results",
        args.optimizer, str(args.dataset), f"seed_{args.seed}",
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── data ─────────────────────────────────────────────────────
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

    # ── descriptor ───────────────────────────────────────────────
    descriptor = GenerateDescriptor(
        max_tree_len=256,
        input_len=input_dim,
        output_len=1,
        using_funcs=OPS,
        max_layer_cnt=8,
        const_samples=[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3],
        layer_leaf_prob=0.3,
    )

    # ── optimizer ────────────────────────────────────────────────
    opt_cls = OPTIMIZER_CLASSES[args.optimizer]
    optimizer = None
    if opt_cls is not None:
        optimizer = opt_cls(
            n_optimize=OPT_N,
            n_offspring=OPT_OFFSPRING,
            opt_iterations=OPT_ITERS,
        )

    # ── train each output dim ────────────────────────────────────
    all_histories = []
    all_summaries = []

    for out_idx in range(output_dim):
        ty = y_train[:, out_idx].contiguous().view(-1, 1)
        vy = y_val[:, out_idx].contiguous()
        t0 = time.time()

        model = Regressor(
            descriptor,
            DefaultCrossover(crossover_rate=CROSSOVER_RATE),
            DefaultMutation(
                mutation_rate=MUTATION_RATE,
                descriptor=descriptor.update(max_layer_cnt=3),
            ),
            TournamentSelection(
                tournament_size=TOURNAMENT_SIZE,
                survivor_rate=SURVIVOR_RATE,
            ),
            pop_size=POP_SIZE,
            elite_rate=ELITE_RATE,
            generation_limit=GENERATION_LIMIT,
            print_mse=True,
            print_mse_prefix=f"  [out{out_idx}] ",
            inject_rate=INJECT_RATE,
            optimizer=optimizer,
            optim_interval=OPT_INTERVAL,
        )
        vy_2d = vy.view(-1, 1)
        model.fit(X_train, ty, X_val=X_val, y_val=vy_2d)
        elapsed = time.time() - t0

        final_train_mse = model.history["train_mse"][-1] if model.history["train_mse"] else float("nan")
        final_val_mse = model.history["val_mse"][-1] if model.history["val_mse"] else float("nan")

        expr = ""
        try:
            expr = str(model.best_tree.to_sympy_expr())
        except Exception:
            pass

        print(f"  out{out_idx}: train_mse={final_train_mse:.6f}  "
              f"val_mse={final_val_mse:.6f}  time={elapsed:.1f}s")

        all_histories.append(model.history)
        all_summaries.append({
            "output_idx": out_idx,
            "train_mse": final_train_mse,
            "val_mse": final_val_mse,
            "elapsed": elapsed,
            "expression": expr,
        })

    # ── save ─────────────────────────────────────────────────────
    np.savez(
        os.path.join(out_dir, "metrics.npz"),
        **{
            f"train_mse_out{i}": np.array(h["train_mse"])
            for i, h in enumerate(all_histories)
        },
        **{
            f"val_mse_out{i}": np.array(h["val_mse"])
            for i, h in enumerate(all_histories)
        },
        **{
            f"avg_tree_size_out{i}": np.array(h["avg_tree_size"])
            for i, h in enumerate(all_histories)
        },
    )

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "optimizer": args.optimizer,
                "summaries": all_summaries,
            },
            f,
            indent=2,
        )

    print(f"  -> saved to {out_dir}")


if __name__ == "__main__":
    main()
