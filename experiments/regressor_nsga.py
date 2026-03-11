"""
比较 TournamentSelection vs ParetoTournamentSelection 的帕累托前沿。

对同一数据集，使用完全相同的配置（相同初始种群）分别跑两种选择算子，
提取帕累托前沿并绘制 2D 对比图（Tree Size vs MSE）+ 超体积指标。

用法: python regressor_nsga.py [dataset_id]
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
    TournamentSelection,
    ParetoTournamentSelection,
)
from evogp.estimators import Regressor
from evogp.core import Forest, GenerateDescriptor

DATASET_ID = 9
if len(sys.argv) > 1:
    DATASET_ID = int(sys.argv[1])

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
MODELS_DIR = os.path.join(ROOT_DIR, "models", "gp_models", str(DATASET_ID), "nsga_cmp")

RED, RESET = "\033[31m", "\033[0m"

# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════


def clone_forest(f: Forest) -> Forest:
    return Forest(
        f.input_len,
        f.output_len,
        f.batch_node_value.clone(),
        f.batch_node_type.clone(),
        f.batch_subtree_size.clone(),
    )


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


def run_two_phase(
    label,
    selection_p1,
    selection_p2,
    descriptor,
    initial_forest,
    X_train,
    ty,
):
    """运行两阶段 GP，返回 Phase2 模型。"""
    print(f"\n{'#'*60}")
    print(f"  {label}")
    print(f"{'#'*60}")

    t0 = time.time()

    # Phase 1
    print(f"\n  === {label} | 阶段1：结构搜索 ===")
    model_p1 = Regressor(
        descriptor,
        DefaultCrossover(crossover_rate=0.9),
        DefaultMutation(
            mutation_rate=0.15, descriptor=descriptor.update(max_layer_cnt=3)
        ),
        selection_p1,
        initial_forest=clone_forest(initial_forest),
        elite_rate=0.1,
        generation_limit=101,
        print_mse=True,
        print_mse_prefix=f"    [{label[:3]}|P1] ",
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
        selection_p2,
        initial_forest=model_p1.algorithm.forest,
        elite_rate=0.2,
        generation_limit=101,
        print_mse=True,
        print_mse_prefix=f"    [{label[:3]}|P2] ",
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


def plot_pareto_comparison(results, save_path):
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
        ax.step(s, m, where="post", label=label, color=color, linewidth=2, alpha=0.85)
        ax.scatter(s, m, color=color, s=30, zorder=5, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Tree Size (number of nodes)", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(f"Pareto Front Comparison (dataset {DATASET_ID})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # ---- 右图：超体积柱状图 ----
    ref_size = max(max(s) for _, s, _, _ in results if len(s) > 0) * 1.1
    ref_mse = max(max(m) for _, _, m, _ in results if len(m) > 0) * 1.1
    ax2 = axes[1]
    labels_hv = []
    hvs = []
    colors = []
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
OPS = {"+": 0.25, "-": 0.15, "*": 0.30, "sin": 0.10, "cos": 0.10, "exp": 0.05, "abs": 0.05}

descriptor = GenerateDescriptor(
    max_tree_len=256,
    input_len=input_dim,
    output_len=1,
    using_funcs=OPS,
    max_layer_cnt=8,
    const_samples=[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3],
    layer_leaf_prob=0.3,
)

# ========== 生成共享初始种群 ==========
POP_SIZE = 100000
torch.manual_seed(42)
torch.cuda.manual_seed(42)
shared_forest = Forest.random_generate(POP_SIZE, descriptor)
print(f"共享初始种群: {shared_forest.pop_size} 棵树")

# ========== 运行对比 ==========
all_pareto = {}

for out_idx in range(output_dim):
    print(f"\n{'='*60}")
    print(f"  输出维度 {out_idx + 1}/{output_dim}")
    print(f"{'='*60}")

    ty = y_train[:, out_idx].contiguous().view(-1, 1)
    vy = y_val[:, out_idx].contiguous()

    # --- A: 标准锦标赛 ---
    model_std, t_std = run_two_phase(
        "Standard",
        TournamentSelection(tournament_size=20, survivor_rate=0.5),
        TournamentSelection(tournament_size=20, survivor_rate=0.2),
        descriptor,
        shared_forest,
        X_train,
        ty,
    )

    # --- B: 帕累托锦标赛 ---
    model_pareto, t_par = run_two_phase(
        "Pareto",
        ParetoTournamentSelection(tournament_size=5, survivor_rate=0.5),
        ParetoTournamentSelection(tournament_size=5, survivor_rate=0.2),
        descriptor,
        shared_forest,
        X_train,
        ty,
    )

    # --- 提取帕累托前沿 ---
    sizes_std, mses_std = extract_pareto_front(model_std)
    sizes_par, mses_par = extract_pareto_front(model_pareto)

    all_pareto[out_idx] = {
        "std": (sizes_std, mses_std),
        "pareto": (sizes_par, mses_par),
    }

    # --- 验证 MSE ---
    for tag, model, elapsed in [("Standard", model_std, t_std),
                                 ("Pareto", model_pareto, t_par)]:
        pred_v = model.predict(X_val)
        v_mse = torch.mean((pred_v - vy.view(-1, 1)) ** 2).item()
        print(f"\n  [{tag}] 验证MSE: {v_mse:.6f}, 帕累托解数: "
              f"{len(all_pareto[out_idx]['std' if tag == 'Standard' else 'pareto'][0])}, "
              f"耗时: {elapsed:.1f}s")

    # --- 打印帕累托对比 ---
    print(f"\n  帕累托前沿对比 (output {out_idx}):")
    print(f"  {'方法':<12} {'#解':>4} {'最小MSE':>12} {'对应Size':>8} {'最小Size':>8} {'对应MSE':>12}")
    print(f"  {'-'*60}")
    for tag, (sz, ms) in [("Standard", (sizes_std, mses_std)),
                           ("Pareto", (sizes_par, mses_par))]:
        if len(sz) > 0:
            best_mse_idx = np.argmin(ms)
            min_size_idx = np.argmin(sz)
            print(f"  {tag:<12} {len(sz):>4} {ms[best_mse_idx]:>12.6f} "
                  f"{sz[best_mse_idx]:>8} {sz[min_size_idx]:>8} {ms[min_size_idx]:>12.6f}")
        else:
            print(f"  {tag:<12}    0          N/A      N/A      N/A          N/A")

    # --- 绘图 ---
    plot_data = [
        ("Standard Tournament", sizes_std, mses_std, "#2196F3"),
        ("Pareto Tournament", sizes_par, mses_par, "#F44336"),
    ]
    plot_pareto_comparison(
        plot_data,
        os.path.join(MODELS_DIR, f"pareto_comparison_out{out_idx}.png"),
    )

# ========== 保存全部结果 ==========
os.makedirs(MODELS_DIR, exist_ok=True)
with open(os.path.join(MODELS_DIR, "comparison_results.pkl"), "wb") as f:
    pickle.dump(all_pareto, f)
print("\n结果已保存")
print("========== 完成 ==========")
