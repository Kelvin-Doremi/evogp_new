#!/usr/bin/env python
"""
Plot experiment results: MSE convergence curves and average tree size.

Usage:
  python experiments/plot_results.py                         # all available results
  python experiments/plot_results.py --optimizers openes de  # compare two optimizers
  python experiments/plot_results.py --datasets 9 165        # only specific datasets
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "data", "figures")

COLORS = {
    "none":   "#888888",
    "cmaes":  "#e41a1c",
    "openes": "#377eb8",
    "snes":   "#4daf4a",
    "xnes":   "#984ea3",
    "de":     "#ff7f00",
    "shade":  "#a65628",
}


def discover():
    """Return {optimizer: {dataset_id: [seed_dirs]}}."""
    results = {}
    if not os.path.isdir(RESULTS_DIR):
        return results
    for opt in sorted(os.listdir(RESULTS_DIR)):
        opt_path = os.path.join(RESULTS_DIR, opt)
        if not os.path.isdir(opt_path):
            continue
        results[opt] = {}
        for ds in sorted(os.listdir(opt_path), key=lambda x: int(x)):
            ds_path = os.path.join(opt_path, ds)
            if not os.path.isdir(ds_path):
                continue
            seed_dirs = sorted(glob.glob(os.path.join(ds_path, "seed_*")))
            if seed_dirs:
                results[opt][int(ds)] = seed_dirs
    return results


def load_curves(seed_dirs, key="best_mse_out0"):
    """Load a metric array from each seed, truncate to min length, stack."""
    arrays = []
    for sd in seed_dirs:
        npz_path = os.path.join(sd, "metrics.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        if key in data:
            arrays.append(data[key])
    if not arrays:
        return None
    min_len = min(len(a) for a in arrays)
    return np.stack([a[:min_len] for a in arrays])


def load_summaries(seed_dirs):
    """Load summary.json from each seed directory."""
    summaries = []
    for sd in seed_dirs:
        path = os.path.join(sd, "summary.json")
        if os.path.exists(path):
            with open(path) as f:
                summaries.append(json.load(f))
    return summaries


def plot_dataset(dataset_id, opt_data, out_idx=0):
    """Create a figure with MSE curve + tree size curve for one dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Dataset {dataset_id}", fontsize=14, fontweight="bold")

    for opt_name, seed_dirs in sorted(opt_data.items()):
        color = COLORS.get(opt_name, "#333333")

        # MSE curve (prefer val_mse, fall back to train_mse)
        # mse = load_curves(seed_dirs, f"val_mse_out{out_idx}")
        mse = load_curves(seed_dirs, f"train_mse_out{out_idx}")

        start_gen = 50

        if mse is None:
            mse = load_curves(seed_dirs, f"train_mse_out{out_idx}")
        if mse is not None:
            gens = np.arange(mse.shape[1])
            mean = mse.mean(axis=0)
            std = mse.std(axis=0)
            axes[0].plot(gens[start_gen:], mean[start_gen:], label=opt_name, color=color, linewidth=1.5)
            axes[0].fill_between(gens[start_gen:], mean[start_gen:] - std[start_gen:], mean[start_gen:] + std[start_gen:], alpha=0.15, color=color)

        # tree size curve
        sizes = load_curves(seed_dirs, f"avg_tree_size_out{out_idx}")
        if sizes is not None:
            gens = np.arange(sizes.shape[1])
            mean = sizes.mean(axis=0)
            std = sizes.std(axis=0)
            axes[1].plot(gens[start_gen:], mean[start_gen:], label=opt_name, color=color, linewidth=1.5)
            axes[1].fill_between(gens[start_gen:], mean[start_gen:] - std[start_gen:], mean[start_gen:] + std[start_gen:], alpha=0.15, color=color)

    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Validation MSE")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Convergence (val MSE)")

    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Avg Tree Size")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Population Avg Tree Size")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_summary_table(all_data):
    """Create a text summary table of final val MSE (mean±std) and avg elapsed time."""
    lines = []
    all_opts = sorted({opt for opt in all_data})
    all_ds = sorted({ds for opt in all_data for ds in all_data[opt]})

    # val_mse table
    lines.append("=== Val MSE (mean±std) ===")
    header = f"{'Dataset':>8s}" + "".join(f"  {opt:>14s}" for opt in all_opts)
    lines.append(header)
    lines.append("-" * len(header))

    for ds in all_ds:
        row = f"{ds:>8d}"
        for opt in all_opts:
            seed_dirs = all_data.get(opt, {}).get(ds, [])
            summaries = load_summaries(seed_dirs)
            if summaries:
                # vals = [s["summaries"][0]["val_mse"] for s in summaries
                vals = [s["summaries"][0]["train_mse"] for s in summaries
                        if s.get("summaries")]
                if vals:
                    row += f"  {np.mean(vals):>8.4f}±{np.std(vals):.4f}"
                    continue
            row += f"  {'N/A':>14s}"
        lines.append(row)

    # elapsed time table
    lines.append("")
    lines.append("=== Avg Elapsed Time (s) ===")
    header2 = f"{'Dataset':>8s}" + "".join(f"  {opt:>14s}" for opt in all_opts)
    lines.append(header2)
    lines.append("-" * len(header2))

    for ds in all_ds:
        row = f"{ds:>8d}"
        for opt in all_opts:
            seed_dirs = all_data.get(opt, {}).get(ds, [])
            summaries = load_summaries(seed_dirs)
            if summaries:
                times = [s["summaries"][0]["elapsed"] for s in summaries
                         if s.get("summaries")]
                if times:
                    row += f"  {np.mean(times):>8.1f}±{np.std(times):>4.1f}"
                    continue
            row += f"  {'N/A':>14s}"
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Plot GP experiment results")
    parser.add_argument("--optimizers", nargs="*", default=None)
    parser.add_argument("--datasets", type=int, nargs="*", default=None)
    args = parser.parse_args()

    all_data = discover()
    if not all_data:
        print(f"No results found in {RESULTS_DIR}")
        return

    if args.optimizers:
        all_data = {k: v for k, v in all_data.items() if k in args.optimizers}

    all_datasets = sorted({ds for opt in all_data for ds in all_data[opt]})
    if args.datasets:
        all_datasets = [d for d in all_datasets if d in args.datasets]

    print(f"Optimizers: {sorted(all_data.keys())}")
    print(f"Datasets:   {all_datasets}")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    for ds in all_datasets:
        opt_data = {}
        for opt, ds_map in all_data.items():
            if ds in ds_map:
                opt_data[opt] = ds_map[ds]

        if not opt_data:
            continue

        fig = plot_dataset(ds, opt_data)
        path = os.path.join(FIGURES_DIR, f"dataset_{ds}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  saved {path}")

    # summary table
    table = plot_summary_table(all_data)
    print(f"\n{table}")
    table_path = os.path.join(FIGURES_DIR, "summary.txt")
    with open(table_path, "w") as f:
        f.write(table)
    print(f"\n  saved {table_path}")


if __name__ == "__main__":
    main()
