"""
两阶段 GP 符号回归：
  阶段1：结构搜索（高交叉、中变异、轻常数优化）
  阶段2：常数精调（低交叉、低变异、重常数优化），继承阶段1种群
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

from evogp.operators import DefaultCrossover, DefaultMutation, TournamentSelection
from evogp.estimators import Regressor
from evogp.core import GenerateDescriptor

RED, RESET = "\033[31m", "\033[0m"


def print_and_save_pareto(model, save_path, prefix=""):
    """提取、打印并保存帕累托前沿。"""
    algo = model.algorithm
    if not getattr(algo, "enable_pareto_front", False):
        return
    pf = algo.pareto_front
    fitness_arr = pf.fitness.cpu().numpy()
    solution = pf.solution
    valid = np.isfinite(fitness_arr) & (fitness_arr > -np.inf)
    n_valid = int(np.sum(valid))

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"fitness": fitness_arr, "solution": solution}, f)

    print(f"\n{prefix}帕累托前沿 ({n_valid} 个解) 已保存到 {save_path}")
    for size in sorted(np.where(valid)[0]):
        fit = fitness_arr[size]
        tree = solution[size]
        try:
            expr = tree.to_sympy_expr()
        except Exception:
            expr = str(tree)
        print(f"  {prefix}规模 {size}: MSE={RED}{-fit:.6f}{RESET}  ->  {expr}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ========== 数据 ==========
print("\n加载数据...")
X_df = pd.read_csv("999_features.csv")
y_df = pd.read_csv("999_targets.csv")
X = X_df.to_numpy(dtype=np.float32)
y = y_df.to_numpy(dtype=np.float32)
print(f"特征: {X.shape}, 目标: {y.shape}")

n_samples = X.shape[0]
n_train = int(n_samples * 0.8)
indices = np.random.RandomState(42).permutation(n_samples)

X_train = torch.FloatTensor(X[indices[:n_train]]).to(device).contiguous()
X_val = torch.FloatTensor(X[indices[n_train:]]).to(device).contiguous()
y_train = torch.FloatTensor(y[indices[:n_train]]).to(device).contiguous()
y_val = torch.FloatTensor(y[indices[n_train:]]).to(device).contiguous()
print(f"训练: {X_train.shape[0]}, 验证: {X_val.shape[0]}")

input_dim = X.shape[1]
output_dim = y.shape[1]

# ========== 算子配置 ==========
OPS = {"+": 0.25, "-": 0.15, "*": 0.30, "sin": 0.10, "cos": 0.10, "exp": 0.05, "abs": 0.05}

# ========== descriptor ==========
descriptor = GenerateDescriptor(
    max_tree_len=256,
    input_len=input_dim,
    output_len=1,
    using_funcs=OPS,
    max_layer_cnt=8,
    const_samples=[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3],
    layer_leaf_prob=0.3,
)

# ========== 训练 ==========
results = []
for out_idx in range(output_dim):
    print(f"\n{'='*60}")
    print(f"  输出维度 {out_idx + 1}/{output_dim}")
    print(f"{'='*60}")

    ty = y_train[:, out_idx].contiguous().view(-1, 1)
    vy = y_val[:, out_idx].contiguous()
    t0 = time.time()

    # --- 阶段1：结构搜索 ---
    print("\n  === 阶段1：结构搜索 ===")
    model_p1 = Regressor(
        descriptor,
        DefaultCrossover(crossover_rate=0.9),
        DefaultMutation(mutation_rate=0.15, descriptor=descriptor.update(max_layer_cnt=3)),
        TournamentSelection(tournament_size=20, survivor_rate=0.5),
        pop_size=50000,
        elite_rate=0.1,
        generation_limit=101,
        print_mse=True,
        print_mse_prefix="    [P1] ",
        enable_pareto_front=True,
        inject_rate=0.05,
        optim_steps=10,
        optim_n=5000,
        optim_offspring=20,
        optim_interval=20,
    )
    model_p1.fit(X_train, ty)
    print_and_save_pareto(
        model_p1,
        f"models/gp_models/pareto_phase1_out{out_idx}.pkl",
        prefix="[P1] ",
    )

    # --- 阶段2：常数精调，继承阶段1种群 ---
    print("\n  === 阶段2：常数精调 ===")
    model_p2 = Regressor(
        descriptor,
        DefaultCrossover(crossover_rate=0.3),
        DefaultMutation(mutation_rate=0.03, descriptor=descriptor.update(max_layer_cnt=3)),
        TournamentSelection(tournament_size=5, survivor_rate=0.2),
        initial_forest=model_p1.algorithm.forest,
        elite_rate=0.2,
        generation_limit=101,
        print_mse=True,
        print_mse_prefix="    [P2] ",
        enable_pareto_front=True,
        optim_steps=50,
        optim_n=2000,
        optim_offspring=50,
        optim_interval=20,
    )
    model_p2.fit(X_train, ty)
    print_and_save_pareto(
        model_p2,
        f"models/gp_models/pareto_phase2_out{out_idx}.pkl",
        prefix="[P2] ",
    )

    elapsed = time.time() - t0
    best = model_p2

    pred_t = best.predict(X_train)
    pred_v = best.predict(X_val)
    t_mse = torch.mean((pred_t - ty) ** 2).item()
    v_mse = torch.mean((pred_v - vy.view(-1, 1)) ** 2).item()
    print(f"\n  训练MSE: {t_mse:.6f}, 验证MSE: {v_mse:.6f}, 耗时: {elapsed:.1f}s")

    try:
        print(f"  表达式: {best.best_tree.to_sympy_expr()}")
    except Exception:
        pass

    results.append(best)

# ========== 保存 ==========
os.makedirs("models/gp_models", exist_ok=True)
with open("models/gp_models/two_phase_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("\n模型已保存")
print("========== 完成 ==========")
