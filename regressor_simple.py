"""
直接使用 GP 拟合 999_features.csv 和 999_targets.csv，不经过神经网络。
"""
import torch
import numpy as np
import pandas as pd
import os

from evogp.operators import DefaultCrossover, DefaultMutation, TournamentSelection
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

# ========== 使用 GP 直接拟合 ==========
print("\n正在使用 GP 拟合...")

gp_models = []

for out_idx in range(output_dim):
    print(f"  输出维度 {out_idx + 1}/{output_dim}...")

    gp_train_X = X_train.contiguous()
    gp_train_y = y_train[:, out_idx].contiguous().view(-1, 1)
    gp_val_X = X_val.contiguous()
    gp_val_y = y_val[:, out_idx].contiguous()

    descriptor = GenerateDescriptor(
        max_tree_len=64,
        input_len=input_dim,
        output_len=1,
        using_funcs=["+", "-", "*", "/", "sin", "cos", "tan"],
        max_layer_cnt=6,
        const_samples=[-2, -1, 0, 1, 2],
        layer_leaf_prob=0.3,
    )

    gp_model = Regressor(
        descriptor=descriptor,
        crossover=DefaultCrossover(),
        mutation=DefaultMutation(
            mutation_rate=0.1, descriptor=descriptor.update(max_layer_cnt=4)
        ),
        selection=TournamentSelection(tournament_size=20),
        pop_size=1000,
        generation_limit=100,
        elite_rate=0.1,
        print_mse=True,
    )
    gp_model.fit(gp_train_X, gp_train_y)

    # 评估
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

# 保存模型
print("\n正在保存 GP 模型...")
os.makedirs("models/gp_models", exist_ok=True)
import pickle

with open("models/gp_models/raw_gp_models.pkl", "wb") as f:
    pickle.dump(gp_models, f)
with open("models/gp_models/complete_raw_gp_model.pkl", "wb") as f:
    pickle.dump(complete_model, f)
print("GP 模型已保存到 models/gp_models/")

print("\n========== 完成！==========")