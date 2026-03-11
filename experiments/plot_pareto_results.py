#!/usr/bin/env python
"""
Plot Pareto front results and compute hypervolume indicators.

Config directories follow the naming convention ``{optimizer}_{selection}``
(e.g. ``openes_pareto``, ``openes_standard``).  Same optimizer shares a
colour; pareto → solid line, standard → dashed line.

For each dataset creates a figure with:
  - Left:  Pareto fronts for every config (merged across seeds)
  - Right: HV bar chart (mean ± std across seeds)

Also produces a text summary table (HV, val-MSE, elapsed time).

Usage:
  python experiments/plot_pareto_results.py
  python experiments/plot_pareto_results.py --configs openes_pareto_k3 openes_standard
  python experiments/plot_pareto_results.py --datasets 9 165
  python experiments/plot_pareto_results.py --use_train_mse
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results_pareto")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "data", "figures_pareto")

OPTIMIZER_COLORS = {
    "none":   "#ff7f00",
    "cmaes":  "#e41a1c",
    "openes": "#377eb8",
    "snes":   "#4daf4a",
    "xnes":   "#984ea3",
    "de":     "#ff7f00",
    "shade":  "#a65628",
}

import re
from matplotlib.colors import to_rgb

PARETO_K_STYLE = {
    "linestyle": "-", "linewidth": 2.2,
    "markersize": 40, "alpha": 0.90, "facecolors": "fill",
}
STANDARD_STYLE = {
    "linestyle": "--", "linewidth": 1.5,
    "markersize": 25, "alpha": 0.50, "facecolors": "none",
}

CONFIG_COLORS = {
    ("none",   "standard"): "#ffb380",
    ("none",   "pareto", 2): "#ff7f00",
    ("none",   "pareto", 3): "#cc6600",
    ("none",   "pareto", 4): "#994c00",
    ("openes", "standard"): "#8bb8e0",
    ("openes", "pareto", 2): "#377eb8",
    ("openes", "pareto", 3): "#2c6593",
    ("openes", "pareto", 4): "#1b3f5c",
    ("snes",   "standard"): "#a3d4a0",
    ("snes",   "pareto", 2): "#4daf4a",
    ("snes",   "pareto", 3): "#3d8c3a",
    ("snes",   "pareto", 4): "#2d6a2b",
    ("xnes",   "standard"): "#c4a5d0",
    ("xnes",   "pareto", 2): "#984ea3",
    ("xnes",   "pareto", 3): "#7a3e83",
    ("xnes",   "pareto", 4): "#5c2e63",
    ("cmaes",  "standard"): "#f09090",
    ("cmaes",  "pareto", 2): "#e41a1c",
    ("cmaes",  "pareto", 3): "#b61416",
    ("cmaes",  "pareto", 4): "#880f11",
    ("de",     "standard"): "#ffb380",
    ("de",     "pareto", 2): "#ff7f00",
    ("de",     "pareto", 3): "#cc6600",
    ("de",     "pareto", 4): "#994c00",
    ("shade",  "standard"): "#d4a87c",
    ("shade",  "pareto", 2): "#a65628",
    ("shade",  "pareto", 3): "#854520",
    ("shade",  "pareto", 4): "#633318",
}

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _auto_color(opt, sel, pk):
    """Generate a colour by darkening the base OPTIMIZER_COLORS entry."""
    base_hex = OPTIMIZER_COLORS.get(opt, "#333333")
    r, g, b = to_rgb(base_hex)
    if sel == "standard":
        f = 1.3
        r = r + (1 - r) * (f - 1)
        g = g + (1 - g) * (f - 1)
        b = b + (1 - b) * (f - 1)
    elif pk > 0:
        f = 1.0 - 0.18 * (pk - 2)
        r, g, b = r * f, g * f, b * f
    return (min(max(r, 0), 1), min(max(g, 0), 1), min(max(b, 0), 1))

_PARETO_K_RE = re.compile(r"^(.+)_pareto_k(\d+)$")


def parse_config(config_name):
    """Split config name → (optimizer, selection_tag, pareto_k).

    Examples:
      'openes_pareto_k3' → ('openes', 'pareto', 3)
      'openes_standard'  → ('openes', 'standard', 0)
      'openes_pareto'    → ('openes', 'pareto', 0)     # legacy
    """
    m = _PARETO_K_RE.match(config_name)
    if m:
        return m.group(1), "pareto", int(m.group(2))
    if config_name.endswith("_standard"):
        return config_name[: -len("_standard")], "standard", 0
    if config_name.endswith("_pareto"):
        return config_name[: -len("_pareto")], "pareto", 0
    return config_name, "unknown", 0


def get_style(opt, sel, pk):
    """Return (style_dict, colour, marker) for a config."""
    sty = STANDARD_STYLE if sel == "standard" else PARETO_K_STYLE

    key = (opt, sel, pk) if sel == "pareto" else (opt, sel)
    color = CONFIG_COLORS.get(key, _auto_color(opt, sel, pk))

    # Assign a unique marker per (sel, k) variant
    if sel == "standard":
        marker = "D"
    else:
        idx = max(pk - 2, 0) % len(MARKERS)
        marker = MARKERS[idx]

    return sty, color, marker


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def discover(results_dir=None):
    """Return {config_name: {dataset_id: [seed_dirs]}}."""
    results_dir = results_dir or RESULTS_DIR
    results = {}
    if not os.path.isdir(results_dir):
        return results
    for cfg in sorted(os.listdir(results_dir)):
        cfg_path = os.path.join(results_dir, cfg)
        if not os.path.isdir(cfg_path):
            continue
        results[cfg] = {}
        for ds in sorted(os.listdir(cfg_path), key=lambda x: int(x)):
            ds_path = os.path.join(cfg_path, ds)
            if not os.path.isdir(ds_path):
                continue
            seed_dirs = sorted(glob.glob(os.path.join(ds_path, "seed_*")))
            if seed_dirs:
                results[cfg][int(ds)] = seed_dirs
    return results


def load_pareto_front(seed_dir, out_idx=0):
    path = os.path.join(seed_dir, f"pareto_front_out{out_idx}.npz")
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path)
    return data["sizes"], data["train_mses"], data["val_mses"]


def load_curves(seed_dirs, key="avg_tree_size_out0"):
    """Load a per-generation metric from each seed, truncate to min length, stack."""
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


def load_summary(seed_dir):
    path = os.path.join(seed_dir, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
# Pareto / HV helpers
# ═══════════════════════════════════════════════════════════════════

def compute_hypervolume_2d(sizes, mses, ref_size, ref_mse):
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


def merge_pareto_fronts(all_sizes, all_mses):
    if not all_sizes:
        return np.array([]), np.array([])
    sizes = np.concatenate(all_sizes)
    mses = np.concatenate(all_mses)

    unique_sizes = np.unique(sizes)
    best_mses = np.array([mses[sizes == s].min() for s in unique_sizes])

    order = np.argsort(unique_sizes)
    unique_sizes, best_mses = unique_sizes[order], best_mses[order]

    p_sizes, p_mses = [], []
    best = float("inf")
    for s, m in zip(unique_sizes, best_mses):
        if m < best:
            p_sizes.append(s)
            p_mses.append(m)
            best = m
    return np.array(p_sizes), np.array(p_mses)


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def _label(opt, sel, pareto_k=0):
    if sel == "pareto" and pareto_k > 0:
        return f"{opt} (k={pareto_k})"
    elif sel == "pareto":
        return f"{opt} (Pareto)"
    return f"{opt} (Std)"


def plot_dataset(dataset_id, cfg_data, use_val=True, out_idx=0):
    """Create figure for one dataset.

    Returns (fig, hv_data) where hv_data = {config_name: [hv_per_seed]}.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(f"Dataset {dataset_id} — Pareto Front Comparison",
                 fontsize=15, fontweight="bold")

    # ── collect per-config Pareto front data ──────────────────────
    per_cfg_pfs = {}
    for cfg_name, seed_dirs in sorted(cfg_data.items()):
        per_cfg_pfs[cfg_name] = []
        for sd in seed_dirs:
            sizes, train_mses, val_mses = load_pareto_front(sd, out_idx)
            if sizes is not None and len(sizes) > 0:
                mses = val_mses if use_val else train_mses
                per_cfg_pfs[cfg_name].append((sizes, mses))

    # ── common reference point for HV ─────────────────────────────
    max_size, max_mse = 0, 0
    for pf_list in per_cfg_pfs.values():
        for sizes, mses in pf_list:
            if len(sizes) > 0:
                max_size = max(max_size, sizes.max())
                max_mse = max(max_mse, mses.max())
    ref_size = max_size * 1.1 if max_size > 0 else 1.0
    ref_mse = max_mse * 1.1 if max_mse > 0 else 1.0

    # ── ordered optimizer list and parsed config info ───────────
    optimizers_ordered = []
    parsed = {}  # cfg_name -> (opt, sel, pareto_k)
    for cfg_name in sorted(cfg_data):
        info = parse_config(cfg_name)
        parsed[cfg_name] = info
        opt = info[0]
        if opt not in optimizers_ordered:
            optimizers_ordered.append(opt)

    # Draw order: standard first (lower zorder), then pareto variants
    draw_order = sorted(cfg_data.keys(),
                        key=lambda c: (0 if parsed[c][1] == "standard" else 1,
                                       parsed[c][2]))

    hv_data = {}
    ax = axes[0]

    # ── draw Pareto fronts ────────────────────────────────────────
    for cfg_name in draw_order:
        opt, sel, pk = parsed[cfg_name]
        sty, color, mkr = get_style(opt, sel, pk)

        pf_list = per_cfg_pfs.get(cfg_name, [])
        if not pf_list:
            continue

        hvs = [compute_hypervolume_2d(s, m, ref_size, ref_mse)
               for s, m in pf_list]
        hv_data[cfg_name] = hvs

        all_s = [s for s, _ in pf_list]
        all_m = [m for _, m in pf_list]
        merged_sizes, merged_mses = merge_pareto_fronts(all_s, all_m)

        if len(merged_sizes) == 0:
            continue

        order = np.argsort(merged_sizes)
        s, m = merged_sizes[order], merged_mses[order]

        ax.step(s, m, where="post",
                label=_label(opt, sel, pk),
                color=color,
                linestyle=sty["linestyle"],
                linewidth=sty["linewidth"],
                alpha=sty["alpha"],
                zorder=3 if sel == "pareto" else 2)

        if sty["facecolors"] == "fill":
            ax.scatter(s, m, s=sty["markersize"], zorder=5,
                       marker=mkr, color=color,
                       edgecolors="white", linewidths=0.6)
        else:
            ax.scatter(s, m, s=sty["markersize"], zorder=5,
                       marker=mkr,
                       facecolors="none", edgecolors=color,
                       linewidths=1.2, alpha=sty["alpha"])

    ylabel = "Validation MSE" if use_val else "Training MSE"
    ax.set_xlabel("Tree Size (nodes)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_yscale("log")
    ax.set_title("Pareto Front (merged across seeds)", fontsize=13)

    ncol = max(1, len(cfg_data) // 4 + 1)
    ax.legend(fontsize=9, ncol=ncol, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── avg tree size per generation ─────────────────────────────
    ax_size = axes[1]
    for cfg_name in draw_order:
        opt, sel, pk = parsed[cfg_name]
        sty, color, _ = get_style(opt, sel, pk)
        seed_dirs = cfg_data[cfg_name]

        curves = load_curves(seed_dirs, f"avg_tree_size_out{out_idx}")
        if curves is None:
            continue

        gens = np.arange(curves.shape[1])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        ax_size.plot(gens, mean,
                     label=_label(opt, sel, pk),
                     color=color,
                     linestyle=sty["linestyle"],
                     linewidth=sty["linewidth"] * 0.7,
                     alpha=sty["alpha"])
        ax_size.fill_between(gens, mean - std, mean + std,
                             color=color, alpha=0.10 if sel == "pareto" else 0.05)

    ax_size.set_xlabel("Generation", fontsize=12)
    ax_size.set_ylabel("Avg Tree Size", fontsize=12)
    ax_size.set_title("Population Avg Tree Size", fontsize=13)
    ncol_s = max(1, len(cfg_data) // 4 + 1)
    ax_size.legend(fontsize=8, ncol=ncol_s, loc="best")
    ax_size.grid(True, alpha=0.3)

    # ── HV grouped bar chart ──────────────────────────────────────
    ax2 = axes[2]

    variant_tags = []
    for cfg_name in draw_order:
        opt, sel, pk = parsed[cfg_name]
        tag = _label("", sel, pk).strip()
        if tag not in variant_tags:
            variant_tags.append(tag)
    n_variants = len(variant_tags)
    bar_width = 0.8 / max(n_variants, 1)
    x = np.arange(len(optimizers_ordered))

    for vidx, tag in enumerate(variant_tags):
        means, stds, bar_colors = [], [], []
        for opt in optimizers_ordered:
            matched = [c for c in draw_order
                       if parsed[c][0] == opt
                       and _label("", parsed[c][1], parsed[c][2]).strip() == tag]
            if matched:
                hvs = hv_data.get(matched[0], [])
                means.append(np.mean(hvs) if hvs else 0)
                stds.append(np.std(hvs) if hvs else 0)
                _, c, _ = get_style(*parsed[matched[0]])
                bar_colors.append(c)
            else:
                means.append(0)
                stds.append(0)
                bar_colors.append("#cccccc")

        offset = (vidx - (n_variants - 1) / 2) * bar_width
        is_std = ("Std" in tag)

        bars = ax2.bar(
            x + offset, means, bar_width,
            yerr=stds, capsize=3,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.8,
            label=tag,
        )
        if is_std:
            for bar in bars:
                bar.set_hatch("///")

        y_off = max(means) * 0.02 if means else 0
        for bar, mv, sv in zip(bars, means, stds):
            if mv > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + sv + y_off,
                    f"{mv:.0f}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax2.set_xticks(x)
    ax2.set_xticklabels(optimizers_ordered, fontsize=10)
    ax2.set_ylabel("Hypervolume (higher = better)", fontsize=12)
    ax2.set_title("HV Indicator (mean ± std)", fontsize=13)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, hv_data


# ═══════════════════════════════════════════════════════════════════
# Summary tables
# ═══════════════════════════════════════════════════════════════════

def build_summary_tables(all_data, all_hv):
    all_cfgs = sorted({k[0] for k in all_hv} | set(all_data.keys()))
    all_ds = sorted({k[1] for k in all_hv})

    col_w = max(len(c) for c in all_cfgs) + 2 if all_cfgs else 16
    col_w = max(col_w, 16)

    lines = []

    # ── HV table ──────────────────────────────────────────────────
    lines.append("=== Hypervolume (mean ± std) ===")
    header = f"{'Dataset':>8s}" + "".join(f"  {c:>{col_w}s}" for c in all_cfgs)
    lines.append(header)
    lines.append("-" * len(header))
    for ds in all_ds:
        row = f"{ds:>8d}"
        for cfg in all_cfgs:
            hvs = all_hv.get((cfg, ds), [])
            if hvs:
                row += f"  {np.mean(hvs):>{col_w - 7}.1f}±{np.std(hvs):>4.1f}"
            else:
                row += f"  {'N/A':>{col_w}s}"
        lines.append(row)

    # ── Val MSE table ─────────────────────────────────────────────
    lines.append("")
    lines.append("=== Best Val MSE (mean ± std) ===")
    lines.append(header)
    lines.append("-" * len(header))
    for ds in all_ds:
        row = f"{ds:>8d}"
        for cfg in all_cfgs:
            seed_dirs = all_data.get(cfg, {}).get(ds, [])
            vals = []
            for sd in seed_dirs:
                s = load_summary(sd)
                if s and s.get("summaries"):
                    vals.append(s["summaries"][0]["val_mse"])
            if vals:
                row += f"  {np.mean(vals):>{col_w - 7}.4f}±{np.std(vals):.4f}"
            else:
                row += f"  {'N/A':>{col_w}s}"
        lines.append(row)

    # ── Elapsed time table ────────────────────────────────────────
    lines.append("")
    lines.append("=== Avg Elapsed Time (s) (mean ± std) ===")
    lines.append(header)
    lines.append("-" * len(header))
    for ds in all_ds:
        row = f"{ds:>8d}"
        for cfg in all_cfgs:
            seed_dirs = all_data.get(cfg, {}).get(ds, [])
            times = []
            for sd in seed_dirs:
                s = load_summary(sd)
                if s and s.get("summaries"):
                    times.append(s["summaries"][0]["elapsed"])
            if times:
                row += f"  {np.mean(times):>{col_w - 5}.1f}±{np.std(times):>4.1f}"
            else:
                row += f"  {'N/A':>{col_w}s}"
        lines.append(row)

    # ── Pareto points table ───────────────────────────────────────
    lines.append("")
    lines.append("=== # Pareto Points (mean ± std) ===")
    lines.append(header)
    lines.append("-" * len(header))
    for ds in all_ds:
        row = f"{ds:>8d}"
        for cfg in all_cfgs:
            seed_dirs = all_data.get(cfg, {}).get(ds, [])
            pts = []
            for sd in seed_dirs:
                s = load_summary(sd)
                if s and s.get("summaries"):
                    pts.append(s["summaries"][0].get("n_pareto_points", 0))
            if pts:
                row += f"  {np.mean(pts):>{col_w - 5}.1f}±{np.std(pts):>4.1f}"
            else:
                row += f"  {'N/A':>{col_w}s}"
        lines.append(row)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Plot Pareto front results and compute HV")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Filter config names (e.g. openes_pareto_k3 de_standard)")
    parser.add_argument("--datasets", type=int, nargs="*", default=None)
    parser.add_argument("--use_train_mse", action="store_true",
                        help="Use training MSE instead of validation MSE")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Override results directory (default: data/results_pareto)")
    parser.add_argument("--figures_dir", type=str, default=None,
                        help="Override figures directory (default: data/figures_pareto)")
    args = parser.parse_args()

    results_dir = args.results_dir or RESULTS_DIR
    figures_dir = args.figures_dir or FIGURES_DIR
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(PROJECT_ROOT, results_dir)
    if not os.path.isabs(figures_dir):
        figures_dir = os.path.join(PROJECT_ROOT, figures_dir)

    all_data = discover(results_dir)
    if not all_data:
        print(f"No results found in {results_dir}")
        return

    if args.configs:
        all_data = {k: v for k, v in all_data.items() if k in args.configs}

    all_datasets = sorted({ds for cfg in all_data for ds in all_data[cfg]})
    if args.datasets:
        all_datasets = [d for d in all_datasets if d in args.datasets]

    use_val = not args.use_train_mse
    print(f"Configs:    {sorted(all_data.keys())}")
    print(f"Datasets:   {all_datasets}")
    print(f"MSE type:   {'validation' if use_val else 'training'}")

    os.makedirs(figures_dir, exist_ok=True)

    all_hv = {}

    for ds in all_datasets:
        cfg_data = {}
        for cfg, ds_map in all_data.items():
            if ds in ds_map:
                cfg_data[cfg] = ds_map[ds]
        if not cfg_data:
            continue

        fig, hv_data = plot_dataset(ds, cfg_data, use_val=use_val)
        path = os.path.join(figures_dir, f"pareto_ds{ds}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")

        for cfg, hvs in hv_data.items():
            all_hv[(cfg, ds)] = hvs

    table = build_summary_tables(all_data, all_hv)
    print(f"\n{table}")
    table_path = os.path.join(figures_dir, "summary.txt")
    with open(table_path, "w") as f:
        f.write(table)
    print(f"\n  saved {table_path}")


if __name__ == "__main__":
    main()
