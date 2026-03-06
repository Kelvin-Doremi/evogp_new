"""加载已保存的 GP 模型或帕累托前沿，评估训练/验证 MSE 并打印表达式。

用法:
    python eval_model.py                                          # 默认加载 two_phase_results.pkl
    python eval_model.py models/gp_models/pareto_phase1_out0.pkl  # 加载帕累托文件
"""
import sys
import pickle

import numpy as np
import pandas as pd
import torch

RED, RESET = "\033[31m", "\033[0m"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# ========== 数据（与训练时完全一致的划分） ==========
X_df = pd.read_csv("999_features.csv")
y_df = pd.read_csv("999_targets.csv")
X = X_df.to_numpy(dtype=np.float32)
y = y_df.to_numpy(dtype=np.float32)

n_samples = X.shape[0]
n_train = int(n_samples * 0.8)
indices = np.random.RandomState(42).permutation(n_samples)

X_train = torch.FloatTensor(X[indices[:n_train]]).to(device).contiguous()
X_val = torch.FloatTensor(X[indices[n_train:]]).to(device).contiguous()
y_train = torch.FloatTensor(y[indices[:n_train]]).to(device).contiguous()
y_val = torch.FloatTensor(y[indices[n_train:]]).to(device).contiguous()

output_dim = y.shape[1]
print(f"特征: {X.shape}, 目标: {y.shape}")
print(f"训练: {X_train.shape[0]}, 验证: {X_val.shape[0]}")

# ========== 加载 ==========
model_path = "models/gp_models/pareto_phase1_out0.pkl"
with open(model_path, "rb") as f:
    data = pickle.load(f)
print(f"\n已加载: {model_path}")


def eval_tree(tree, out_idx):
    """评估单棵树的训练/验证 MSE。"""
    ty = y_train[:, out_idx].contiguous().view(-1, 1)
    vy = y_val[:, out_idx].contiguous()
    pred_t = tree.forward(X_train)
    pred_v = tree.forward(X_val)
    t_mse = torch.mean((pred_t - ty) ** 2).item()
    v_mse = torch.mean((pred_v - vy.view(-1, 1)) ** 2).item()
    return t_mse, v_mse


# ========== 帕累托文件: {"fitness": ..., "solution": ...} ==========
if isinstance(data, dict) and "fitness" in data and "solution" in data:
    fitness_arr = data["fitness"]
    solution = data["solution"]
    valid = np.isfinite(fitness_arr) & (fitness_arr > -np.inf)
    n_valid = int(np.sum(valid))
    print(f"帕累托前沿: {n_valid} 个有效解\n")

    out_idx = 0
    for size in sorted(np.where(valid)[0]):
        tree = solution[size]
        t_mse, v_mse = eval_tree(tree, out_idx)
        try:
            expr = tree.to_sympy_expr()
        except Exception:
            expr = str(tree)
        print(
            f"  规模 {size:3d}: "
            f"训练MSE={RED}{t_mse:.6f}{RESET}  "
            f"验证MSE={RED}{v_mse:.6f}{RESET}  "
            f"->  {expr}"
        )

# ========== 结果文件: [Regressor, ...] ==========
elif isinstance(data, list):
    print(f"{len(data)} 个模型\n")
    for out_idx, model in enumerate(data):
        print(f"{'='*60}")
        print(f"  输出维度 {out_idx + 1}/{output_dim}")
        print(f"{'='*60}")

        t_mse, v_mse = eval_tree(model.best_tree, out_idx)
        print(f"  训练MSE: {RED}{t_mse:.6f}{RESET}")
        print(f"  验证MSE: {RED}{v_mse:.6f}{RESET}")

        try:
            print(f"  表达式: {model.best_tree.to_sympy_expr()}")
        except Exception:
            pass

else:
    print(f"未知文件格式: {type(data)}")
    sys.exit(1)

print(f"\n{'='*60}")
print("评估完成")
