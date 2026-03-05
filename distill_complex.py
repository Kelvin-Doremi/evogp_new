"""
直接使用 GP 拟合 999_features.csv 和 999_targets.csv，不经过神经网络。
"""
import torch
import numpy as np
import pandas as pd
import os

from evogp.operators import (
    DefaultCrossover,
    DefaultMutation,
    TournamentSelection,
)
from evogp.estimators import Regressor
from evogp.core import Forest, GenerateDescriptor

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")

# ========== 加载数据 ==========
print("\n========== 直接 GP 拟合 CSV 数据 ==========")
print("\n正在加载 999_features.csv 和 999_targets.csv...")

X_df = pd.read_csv("999_features.csv")
y_df = pd.read_csv("999_targets.csv")

X = X_df.to_numpy(dtype=np.float32)
y = y_df.to_numpy(dtype=np.float32)

print(f"特征形状: {X.shape}, 目标形状: {y.shape}")
print(f"输入维度: {X.shape[1]}, 输出维度: {y.shape[1]}")

# 划分训练集和验证集（80/20）
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

# ========== 第一阶段：结构搜索（高交叉、中变异、强选择压力）==========
print("\n========== 第一阶段：结构搜索 200 代 ==========")

gp_models = []
descriptor = GenerateDescriptor(
    max_tree_len=32,
    input_len=input_dim,
    output_len=1,
    using_funcs=["+", "-", "*", "/", "sin", "cos", "tan"],
    max_layer_cnt=5,
    const_samples=[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3],
    layer_leaf_prob=0.3,
)

for out_idx in range(output_dim):
    print(f"  输出维度 {out_idx + 1}/{output_dim}...")

    gp_train_X = X_train.contiguous()
    gp_train_y = y_train[:, out_idx].contiguous().view(-1, 1)
    gp_val_X = X_val.contiguous()
    gp_val_y = y_val[:, out_idx].contiguous()

    gp_model = Regressor(
        initial_forest=Forest.random_generate(pop_size=10000, descriptor=descriptor),
        crossover=DefaultCrossover(crossover_rate=0.9),
        mutation=DefaultMutation(
            mutation_rate=0.15, descriptor=descriptor.update(max_layer_cnt=3)
        ),
        selection=TournamentSelection(tournament_size=7, survivor_rate=0.3),
        elite_rate=0.05,
        generation_limit=50,
        print_mse=True,
    )
    gp_model.fit(gp_train_X, gp_train_y)

    gp_pred_train = gp_model.predict(gp_train_X)
    gp_pred_val = gp_model.predict(gp_val_X)
    train_mse = torch.mean((gp_pred_train - gp_train_y) ** 2).item()
    val_mse = torch.mean((gp_pred_val - gp_val_y.view(-1, 1)) ** 2).item()
    print(f"    训练MSE: {train_mse:.6f}, 验证MSE: {val_mse:.6f}")

    gp_models.append(gp_model)
    if output_dim == 1:
        try:
            print(f"    最佳表达式: {gp_model.best_tree.to_sympy_expr()}")
        except Exception:
            pass

# ========== 第二阶段：交替优化（重常数优化 ↔ 轻结构探索）==========
#
# 思路类似 NAS + Training 的交替范式：
#   1) 固定结构，重度优化常数 → 让好结构真正"发光"
#   2) 低变异、高精英率，轻度探索结构 → 修复/改进结构
#   3) 循环往复
# 这样不会完全冻结结构（避免被差结构卡住），
# 也不会每代都大改结构（避免破坏已优化的常数）。
#
import time
import copy
from evox.algorithms import CMAES, DE, PSO, CSO
from evogp.operators.optimization import ConstantOptimization, evox_factory
from evogp.workflows import GeneticProgramming

algorithms_to_test = {
    "CMA-ES": evox_factory(CMAES, sigma=1.0),
    "DE":  evox_factory(DE,  lb=-10, ub=10),
    "PSO": evox_factory(PSO, lb=-10, ub=10),
    "CSO": evox_factory(CSO, lb=-10, ub=10),
}

phase1_forest = gp_models[0].algorithm.forest

light_crossover = DefaultCrossover(crossover_rate=0.3)
light_mutation = DefaultMutation(
    mutation_rate=0.03, descriptor=descriptor.update(max_layer_cnt=3)
)
light_selection = TournamentSelection(tournament_size=5, survivor_rate=0.2)
LIGHT_ELITE_RATE = 0.2


class TreeModel:
    """Wraps a single Tree with a predict() interface."""

    def __init__(self, best_tree):
        self.best_tree = best_tree

    def predict(self, X):
        return self.best_tree.forward(X)


N_CYCLES = 6
GP_GENS_PER_CYCLE = 5

algo_results = {}
for algo_name, factory in algorithms_to_test.items():
    print(f"\n========== 第二阶段：交替优化 + {algo_name} ==========")

    for out_idx in range(output_dim):
        print(f"  输出维度 {out_idx + 1}/{output_dim}...")

        gp_train_X = X_train.contiguous()
        gp_train_y = y_train[:, out_idx].contiguous().view(-1, 1)
        gp_val_X = X_val.contiguous()
        gp_val_y = y_val[:, out_idx].contiguous()

        forest_copy = copy.deepcopy(phase1_forest)

        t0 = time.time()
        for cycle in range(N_CYCLES):
            # --- Step A: 重度常数优化（结构不变）---
            opt = ConstantOptimization(
                n_optimize=10,
                n_offspring=100,
                opt_iterations=100,
                algorithm_factory=factory,
            )
            fitnesses = opt(forest_copy, gp_train_X, gp_train_y)
            best_mse = -torch.max(fitnesses).item()
            print(
                f"    Cycle {cycle + 1}/{N_CYCLES} "
                f"常数优化: best MSE = {best_mse:.6f}"
            )

            # --- Step B: 轻度结构探索（最后一轮跳过）---
            if cycle < N_CYCLES - 1:
                gp = GeneticProgramming(
                    forest_copy,
                    light_crossover,
                    light_mutation,
                    light_selection,
                    elite_rate=LIGHT_ELITE_RATE,
                )
                for _ in range(GP_GENS_PER_CYCLE):
                    gp.step(fitnesses)
                    fitnesses = -gp.forest.SR_fitness(gp_train_X, gp_train_y)
                    fitnesses[torch.isnan(fitnesses)] = -torch.inf
                forest_copy = gp.forest
                best_mse_gp = -torch.max(fitnesses).item()
                print(
                    f"    Cycle {cycle + 1}/{N_CYCLES} "
                    f"结构探索: best MSE = {best_mse_gp:.6f}"
                )

        elapsed = time.time() - t0

        best_idx = torch.argmax(fitnesses).item()
        best_tree = forest_copy[best_idx]

        pred_train = best_tree.forward(gp_train_X)
        pred_val = best_tree.forward(gp_val_X)
        train_mse = torch.mean(
            (pred_train.view(-1) - gp_train_y.view(-1)) ** 2
        ).item()
        val_mse = torch.mean(
            (pred_val.view(-1) - gp_val_y.view(-1)) ** 2
        ).item()
        print(
            f"    训练MSE: {train_mse:.6f}, 验证MSE: {val_mse:.6f}, "
            f"耗时: {elapsed:.1f}s"
        )

        algo_results[algo_name] = {
            "train_mse": train_mse,
            "val_mse": val_mse,
            "time": elapsed,
            "model": TreeModel(best_tree),
        }

        if output_dim == 1:
            try:
                print(f"    最佳表达式: {best_tree.to_sympy_expr()}")
            except Exception:
                pass

# 对比结果汇总
print("\n========== 算法对比结果 ==========")
print(f"{'算法':<8} {'训练MSE':<14} {'验证MSE':<14} {'耗时':<10}")
print("-" * 46)
for name, res in algo_results.items():
    print(
        f"{name:<8} {res['train_mse']:<14.6f} "
        f"{res['val_mse']:<14.6f} {res['time']:<10.1f}s"
    )

best_algo = min(algo_results, key=lambda k: algo_results[k]["val_mse"])
print(f"\n最佳算法: {best_algo} (验证MSE={algo_results[best_algo]['val_mse']:.6f})")
gp_models[0] = algo_results[best_algo]["model"]

# ========== 整合模型并评估 ==========
print("\n评估完整 GP 模型...")


class CompleteGPModel:
    """整合多个输出维度的 GP 模型"""

    def __init__(self, gp_models):
        self.gp_models = gp_models

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X).to(device)
        X = X.contiguous()

        outputs = []
        for gp_model in self.gp_models:
            pred = gp_model.predict(X)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)
            outputs.append(pred)
        return torch.cat(outputs, dim=1).contiguous()


complete_model = CompleteGPModel(gp_models)

train_pred = complete_model.predict(X_train)
val_pred = complete_model.predict(X_val)
train_mse_final = torch.mean((train_pred - y_train) ** 2).item()
val_mse_final = torch.mean((val_pred - y_val) ** 2).item()

print(f"完整 GP 模型 - 训练MSE: {train_mse_final:.6f}, 验证MSE: {val_mse_final:.6f}")

# 保存模型（跳过不可序列化的 evox 闭包）
print("\n正在保存 GP 模型...")
os.makedirs("models/gp_models", exist_ok=True)
try:
    import pickle
    with open("models/gp_models/raw_gp_models.pkl", "wb") as f:
        pickle.dump(gp_models, f)
    with open("models/gp_models/complete_raw_gp_model.pkl", "wb") as f:
        pickle.dump(complete_model, f)
    print("GP 模型已保存到 models/gp_models/")
except Exception as e:
    print(f"保存跳过（{e}）")

print("\n========== 完成！==========")