"""
直接使用 GP 拟合 999_features.csv 和 999_targets.csv，不经过神经网络。
两阶段训练：
  阶段1：结构搜索（高交叉、中变异、轻常数优化）
  阶段2：交替优化（低交叉、低变异、重常数优化），继承阶段1种群
"""

import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

from evogp.operators import DefaultCrossover, DefaultMutation, TournamentSelection
from evogp.estimators import Regressor, BoostedRegressor
from evogp.core import Forest, GenerateDescriptor

# ========== 工具函数 ==========

RED, RESET = "\033[31m", "\033[0m"


def get_pareto(model):
    """从 Regressor 的 GeneticProgramming 中提取帕累托前沿。"""
    algo = model.algorithm
    if not getattr(algo, "enable_pareto_front", False):
        return None
    pf = algo.pareto_front
    return {"fitness": pf.fitness.cpu().numpy(), "solution": pf.solution}


def save_and_print_pareto(pareto_results, save_path, prefix=""):
    """保存帕累托前沿到文件并打印各解（仅打印有有效 fitness 的）。"""
    if not pareto_results or all(p is None for p in pareto_results):
        return
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(pareto_results, f)
    print(f"\n{prefix}帕累托前沿已保存到 {save_path}")
    for out_idx, pr in enumerate(pareto_results):
        if pr is None:
            continue
        fitness_arr = pr["fitness"]
        solution_forest = pr["solution"]
        valid = np.isfinite(fitness_arr) & (fitness_arr > -np.inf)
        n_valid = int(np.sum(valid))
        print(f"\n{prefix}输出维度 {out_idx}: {n_valid} 个不同规模的帕累托解")
        for size in sorted(np.where(valid)[0]):
            fit = fitness_arr[size]
            mse = -fit
            tree = solution_forest[size]
            try:
                expr = tree.to_sympy_expr()
            except Exception:
                expr = str(tree)
            print(
                f"  {prefix}规模 {size}: fitness={RED}{fit:.6f}{RESET} "
                f"\n ->  {expr}"
            )


# ========== 设备 & 数据 ==========

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")

print("\n正在加载 999_features.csv 和 999_targets.csv...")
X_df = pd.read_csv("999_features.csv")
y_df = pd.read_csv("999_targets.csv")

X = X_df.to_numpy(dtype=np.float32)
y = y_df.to_numpy(dtype=np.float32)
print(f"特征形状: {X.shape}, 目标形状: {y.shape}")

n_samples = X.shape[0]
n_train = int(n_samples * 0.8)
indices = np.random.RandomState(42).permutation(n_samples)
train_idx, val_idx = indices[:n_train], indices[n_train:]

X_train = torch.FloatTensor(X[train_idx]).to(device).contiguous()
X_val = torch.FloatTensor(X[val_idx]).to(device).contiguous()
y_train = torch.FloatTensor(y[train_idx]).to(device).contiguous()
y_val = torch.FloatTensor(y[val_idx]).to(device).contiguous()
print(f"训练集: {X_train.shape[0]} 样本, 验证集: {X_val.shape[0]} 样本")

input_dim = X.shape[1]
output_dim = y.shape[1]

# ========== 算子配置 & descriptor ==========

OPS = {"+": 0.25, "-": 0.15, "*": 0.30, "sin": 0.10, "cos": 0.10, "exp": 0.05, "abs": 0.05}

descriptor = GenerateDescriptor(
    max_tree_len=64,
    input_len=input_dim,
    output_len=1,
    using_funcs=OPS,
    max_layer_cnt=6,
    const_samples=[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3],
    layer_leaf_prob=0.3,
)

# ========== 训练（BoostedRegressor）==========

N_TERMS = 4

gp_models = []
for out_idx in range(output_dim):
    print(f"\n{'='*60}")
    print(f"  输出维度 {out_idx + 1}/{output_dim} — Boosted GP ({N_TERMS} 项)")
    print(f"{'='*60}")

    gp_train_X = X_train.contiguous()
    gp_train_y = y_train[:, out_idx].contiguous().view(-1, 1)
    gp_val_X = X_val.contiguous()
    gp_val_y = y_val[:, out_idx].contiguous()

    t0 = time.time()
    model = BoostedRegressor(
        n_terms=N_TERMS,
        descriptor=descriptor,
        crossover=DefaultCrossover(crossover_rate=0.9),
        mutation=DefaultMutation(
            mutation_rate=0.15, descriptor=descriptor.update(max_layer_cnt=3)
        ),
        selection=TournamentSelection(tournament_size=20, survivor_rate=0.5),
        pop_size=50000,
        regressor_kwargs=dict(
            elite_rate=0.1,
            generation_limit=101,
            print_mse=True,
            enable_pareto_front=True,
            inject_rate=0.05,
            optim_steps=10,
            optim_n=5000,
            optim_offspring=20,
            optim_interval=20,
        ),
        phase2_kwargs=dict(
            crossover=DefaultCrossover(crossover_rate=0.3),
            mutation=DefaultMutation(
                mutation_rate=0.03, descriptor=descriptor.update(max_layer_cnt=3)
            ),
            selection=TournamentSelection(tournament_size=5, survivor_rate=0.2),
            elite_rate=0.2,
            generation_limit=101,
            print_mse=True,
            optim_steps=50,
            optim_n=2000,
            optim_offspring=50,
            optim_interval=20,
        ),
    )
    model.fit(gp_train_X, gp_train_y)
    elapsed = time.time() - t0

    pred_train = model.predict(gp_train_X)
    pred_val = model.predict(gp_val_X)
    if pred_train.dim() > 1:
        pred_train = pred_train.view(-1)
    if pred_val.dim() > 1:
        pred_val = pred_val.view(-1)
    train_mse = torch.mean((pred_train - gp_train_y.view(-1)) ** 2).item()
    val_mse = torch.mean((pred_val - gp_val_y.view(-1)) ** 2).item()
    print(f"\n  训练MSE: {train_mse:.6f}, 验证MSE: {val_mse:.6f}, 耗时: {elapsed:.1f}s")
    print(f"  完整表达式: {model.get_sympy_expr()}")

    gp_models.append(model)

# ========== 整合多输出维度并评估 ==========
print("\n评估完整 Boosted GP 模型...")


class CompleteBoostedModel:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X).to(device)
        X = X.contiguous()
        outputs = []
        for m in self.models:
            pred = m.predict(X)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)
            elif pred.dim() == 0:
                pred = pred.unsqueeze(0).unsqueeze(1)
            outputs.append(pred)
        return torch.cat(outputs, dim=1).contiguous()


complete_model = CompleteBoostedModel(gp_models)

train_pred = complete_model.predict(X_train)
val_pred = complete_model.predict(X_val)
train_mse_final = torch.mean((train_pred - y_train) ** 2).item()
val_mse_final = torch.mean((val_pred - y_val) ** 2).item()
print(f"完整模型 - 训练MSE: {train_mse_final:.6f}, 验证MSE: {val_mse_final:.6f}")

# ========== 保存模型 ==========
print("\n正在保存模型...")
os.makedirs("models/gp_models", exist_ok=True)
try:
    with open("models/gp_models/boosted_gp_models.pkl", "wb") as f:
        pickle.dump(gp_models, f)
    with open("models/gp_models/complete_boosted_model.pkl", "wb") as f:
        pickle.dump(complete_model, f)
    print("模型已保存到 models/gp_models/")
except Exception as e:
    print(f"保存跳过（{e}）")

print("\n========== 完成！==========")
