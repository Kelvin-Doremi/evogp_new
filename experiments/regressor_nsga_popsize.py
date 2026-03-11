"""
比较 大种群 vs 小种群 Pareto Tournament 的帕累托前沿。

同一数据集、同一选择算子（ParetoTournamentSelection），
仅种群规模不同，对比帕累托前沿质量。

用法: python regressor_nsga_popsize.py [dataset_id]
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evogp.operators import (
    DefaultCrossover,
    DefaultMutation,
    ParetoTournamentSelection,
)
from evogp.estimators import Regressor
from evogp.core import Forest, GenerateDescriptor

DATASET_ID = 165
if len(sys.argv) > 1:
    DATASET_ID = int(sys.argv[1])

# 大种群 vs 小种群
POP_LARGE = 50000
POP_SMALL = 5000

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
MODELS_DIR = os.path.join(
    ROOT_DIR, "models", "gp_models", str(DATASET_ID), "popsize_cmp"
)


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════


def extract_pareto_front(model):
    """从 Regressor 提取帕累托前沿 → (sizes, mses)，已去除被支配的点。"""
    algo = model.algorithm
    if not getattr(algo, "enable_pareto_front", False):
        return np.array([]), np.array([])
    pf = algo.pareto_front
    fitness_arr = pf.fitness.cpu().numpy()
    valid = np.isfinite(fitness_arr) & (fitness_arr > -np.inf)
    sizes = np.where(valid)[0]
    mses = -fitness_arr[sizes]

    pareto_sizes, pareto_mses = [], []
    best_mse = float("inf")
    for s, m in zip(sizes, mses):
        if m < best_mse:
            pareto_sizes.append(s)
            pareto_mses.append(m)
            best_mse = m
    return np.array(pareto_sizes), np.array(pareto_mses)


def compute_hypervolume_2d(sizes, mses, ref_size, ref_mse):
    """2D 超体积：帕累托前沿台阶曲线与参考点围成的面积。"""
    if len(sizes) == 0:
        return 0.0
    order = np.argsort(sizes)
    sizes, mses = sizes[order], mses[order]
    hv = 0.0
    for i in range(len(sizes)):
        right = sizes[i + 1] if i + 1 < len(sizes) else ref_size
        height = ref_mse - mses[i]
        if height > 0 and right > sizes[i]:
            hv += (right - sizes[i]) * height
    return hv


def run_two_phase(label, pop_size, descriptor, X_train, ty, seed=42):
    """运行两阶段 GP（Pareto Tournament），返回 Phase2 模型。"""
    print(f"\n{'#'*60}")
    print(f"  {label}  (pop_size={pop_size})")
    print(f"{'#'*60}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    t0 = time.time()

    tag = f"{label[:5]}"

    # Phase 1
    print(f"\n  === {label} | 阶段1：结构搜索 ===")
    model_p1 = Regressor(
        descriptor,
        DefaultCrossover(crossover_rate=0.9),
        DefaultMutation(
            mutation_rate=0.15, descriptor=descriptor.update(max_layer_cnt=3)
        ),
        ParetoTournamentSelection(tournament_size=3, survivor_rate=0.5),
        pop_size=pop_size,
        elite_rate=0.1,
        generation_limit=101,
        print_mse=True,
        print_mse_prefix=f"    [{tag}|P1] ",
        enable_pareto_front=True,
        inject_rate=0.05,
        optim_steps=0,
        optim_n=5000,
        optim_offspring=20,
        optim_interval=20,
    )
    model_p1.fit(X_train, ty)

    # Phase 2
    print(f"\n  === {label} | 阶段2：常数精调 ===")
    model_p2 = Regressor(
        descriptor,
        DefaultCrossover(crossover_rate=0.3),
        DefaultMutation(
            mutation_rate=0.03, descriptor=descriptor.update(max_layer_cnt=3)
        ),
        ParetoTournamentSelection(tournament_size=3, survivor_rate=0.2),
        initial_forest=model_p1.algorithm.forest,
        elite_rate=0.2,
        generation_limit=101,
        print_mse=True,
        print_mse_prefix=f"    [{tag}|P2] ",
        enable_pareto_front=True,
        optim_steps=0,
        optim_n=2000,
        optim_offspring=50,
        optim_interval=20,
    )
    model_p2.fit(X_train, ty)

    elapsed = time.time() - t0
    print(f"\n  {label} 总耗时: {elapsed:.1f}s")
    return model_p2, elapsed


def plot_pareto_comparison(results, save_path, title_suffix=""):
    """绘制帕累托前沿对比图。

    results: list of (label, sizes, mses, color)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---- 左图：帕累托前沿台阶图 ----
    ax = axes[0]
    for label, sizes, mses, color in results:
        if len(sizes) == 0:
            continue
        order = np.argsort(sizes)
        s, m = sizes[order], mses[order]
        ax.step(
            s, m, where="post", label=label, color=color, linewidth=2, alpha=0.85
        )
        ax.scatter(
            s, m, color=color, s=30, zorder=5, edgecolors="white", linewidths=0.5
        )
    ax.set_xlabel("Tree Size (number of nodes)", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(
        f"Pareto Front: Large vs Small Pop (dataset {DATASET_ID}){title_suffix}",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # ---- 右图：超体积柱状图 ----
    non_empty = [(l, s, m, c) for l, s, m, c in results if len(s) > 0]
    if non_empty:
        ref_size = max(max(s) for _, s, _, _ in non_empty) * 1.1
        ref_mse = max(max(m) for _, _, m, _ in non_empty) * 1.1
    else:
        ref_size, ref_mse = 1.0, 1.0

    ax2 = axes[1]
    labels_hv, hvs, colors = [], [], []
    for label, sizes, mses, color in results:
        hv = compute_hypervolume_2d(sizes, mses, ref_size, ref_mse)
        labels_hv.append(label)
        hvs.append(hv)
        colors.append(color)
    bars = ax2.bar(labels_hv, hvs, color=colors, alpha=0.8, edgecolor="black")
    for bar, hv in zip(bars, hvs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{hv:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax2.set_ylabel("Hypervolume (higher = better)", fontsize=12)
    ax2.set_title("Hypervolume Indicator", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n对比图已保存: {save_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ========== 数据 ==========
print(f"\n加载数据 (dataset_id={DATASET_ID})...")
X_df = pd.read_csv(os.path.join(DATASETS_DIR, f"{DATASET_ID}_features.csv"))
y_df = pd.read_csv(os.path.join(DATASETS_DIR, f"{DATASET_ID}_targets.csv"))
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
OPS = {
    "+": 0.25, "-": 0.15, "*": 0.30,
    "sin": 0.10, "cos": 0.10, "exp": 0.05, "abs": 0.05,
}

descriptor = GenerateDescriptor(
    max_tree_len=256,
    input_len=input_dim,
    output_len=1,
    using_funcs=OPS,
    max_layer_cnt=8,
    const_samples=[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3],
    layer_leaf_prob=0.3,
)

# ========== 运行对比 ==========
all_pareto = {}

for out_idx in range(output_dim):
    print(f"\n{'='*60}")
    print(f"  输出维度 {out_idx + 1}/{output_dim}")
    print(f"{'='*60}")

    ty = y_train[:, out_idx].contiguous().view(-1, 1)
    vy = y_val[:, out_idx].contiguous()

    # --- A: 大种群 ---
    model_large, t_large = run_two_phase(
        f"Large({POP_LARGE})", POP_LARGE, descriptor, X_train, ty, seed=42
    )

    # --- B: 小种群 ---
    model_small, t_small = run_two_phase(
        f"Small({POP_SMALL})", POP_SMALL, descriptor, X_train, ty, seed=42
    )

    # --- 提取帕累托前沿 ---
    sizes_l, mses_l = extract_pareto_front(model_large)
    sizes_s, mses_s = extract_pareto_front(model_small)

    all_pareto[out_idx] = {
        "large": (sizes_l, mses_l, t_large),
        "small": (sizes_s, mses_s, t_small),
    }

    # --- 验证 MSE ---
    for tag, model, elapsed in [
        (f"Large({POP_LARGE})", model_large, t_large),
        (f"Small({POP_SMALL})", model_small, t_small),
    ]:
        pred_v = model.predict(X_val)
        v_mse = torch.mean((pred_v - vy.view(-1, 1)) ** 2).item()
        key = "large" if "Large" in tag else "small"
        n_sol = len(all_pareto[out_idx][key][0])
        print(f"\n  [{tag}] 验证MSE: {v_mse:.6f}, 帕累托解数: {n_sol}, 耗时: {elapsed:.1f}s")

    # --- 打印帕累托对比 ---
    print(f"\n  帕累托前沿对比 (output {out_idx}):")
    print(
        f"  {'方法':<20} {'#解':>4} {'最小MSE':>12} {'对应Size':>8} "
        f"{'最小Size':>8} {'对应MSE':>12} {'耗时(s)':>8}"
    )
    print(f"  {'-'*76}")
    for tag, key in [(f"Large({POP_LARGE})", "large"), (f"Small({POP_SMALL})", "small")]:
        sz, ms, elapsed = all_pareto[out_idx][key]
        if len(sz) > 0:
            best_mse_idx = np.argmin(ms)
            min_size_idx = np.argmin(sz)
            print(
                f"  {tag:<20} {len(sz):>4} {ms[best_mse_idx]:>12.6f} "
                f"{sz[best_mse_idx]:>8} {sz[min_size_idx]:>8} "
                f"{ms[min_size_idx]:>12.6f} {elapsed:>8.1f}"
            )
        else:
            print(f"  {tag:<20}    0          N/A      N/A      N/A          N/A      N/A")

    # --- 绘图 ---
    plot_data = [
        (f"Pareto Large (pop={POP_LARGE})", sizes_l, mses_l, "#2196F3"),
        (f"Pareto Small (pop={POP_SMALL})", sizes_s, mses_s, "#FF9800"),
    ]
    plot_pareto_comparison(
        plot_data,
        os.path.join(MODELS_DIR, f"popsize_comparison_out{out_idx}.png"),
    )

# ========== 保存全部结果 ==========
os.makedirs(MODELS_DIR, exist_ok=True)
with open(os.path.join(MODELS_DIR, "popsize_comparison_results.pkl"), "wb") as f:
    pickle.dump(all_pareto, f)
print("\n结果已保存")
print("========== 完成 ==========")
