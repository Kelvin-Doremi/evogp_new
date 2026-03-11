#!/usr/bin/env python
"""
Launch GP experiments for Pareto front comparison across GPUs.

Each job is a (dataset, seed, optimizer, selection, pareto_tsize) combo.
By default, both selection strategies (pareto + standard) are run for
every optimizer.  For pareto selection, multiple tournament sizes can be
swept via --pareto_tsizes, producing configs like ``openes_pareto_k2``.

Usage:
  python experiments/run_all_pareto.py --optimizers openes de shade
  python experiments/run_all_pareto.py --optimizers none openes --pareto_tsizes 2 3 4
  python experiments/run_all_pareto.py --optimizers openes --n_gpus 4 --n_seeds 2
  python experiments/run_all_pareto.py --optimizers openes --datasets 9 165
"""
import argparse
import glob
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(PROJECT_ROOT, "experiments", "regressor_pareto.py")

# ALL_OPTIMIZERS = ["none", "cmaes", "openes", "snes", "xnes", "de", "shade"]
ALL_OPTIMIZERS = ["none", "openes", "shade"]
ALL_SELECTIONS = ["pareto", "standard"]


def discover_datasets():
    """Auto-detect dataset IDs from data/datasets/."""
    pattern = os.path.join(PROJECT_ROOT, "data", "datasets", "*_features.csv")
    return sorted(
        int(os.path.basename(p).split("_")[0])
        for p in glob.glob(pattern)
    )


def run_batch(jobs, n_gpus):
    """Run jobs with at most n_gpus concurrent (explicit free-GPU pool)."""
    pending = list(jobs)
    free_gpus = list(range(n_gpus))
    running = {}  # gpu_id -> (proc, desc)

    while pending or running:
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            dataset, seed, opt, sel, ptsize = pending.pop(0)
            cmd = [
                sys.executable, SCRIPT,
                "--dataset", str(dataset),
                "--seed", str(seed),
                "--gpu", str(gpu),
                "--optimizer", opt,
                "--selection", sel,
                "--pareto_tsize", str(ptsize),
            ]
            desc = f"ds={dataset} seed={seed} opt={opt} sel={sel} k={ptsize} gpu={gpu}"
            print(f"  LAUNCH  {desc}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            running[gpu] = (proc, desc)

        for gpu in list(running):
            proc, desc = running[gpu]
            ret = proc.poll()
            if ret is not None:
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"  {status:8s}  {desc}")
                if ret != 0:
                    stderr = proc.stderr.read().decode()[-500:]
                    print(f"           stderr: {stderr}")
                del running[gpu]
                free_gpus.append(gpu)

        if running:
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(
        description="Run Pareto-front comparison GP experiments")
    parser.add_argument(
        "--optimizers", type=str, nargs="+", required=True,
        help=f"Optimizers to run. Choices: {', '.join(ALL_OPTIMIZERS)}",
    )
    parser.add_argument(
        "--selections", type=str, nargs="+", default=ALL_SELECTIONS,
        help="Selection strategies (default: pareto standard)",
    )
    parser.add_argument(
        "--pareto_tsizes", type=int, nargs="+", default=[2, 3, 4],
        help="Pareto tournament sizes to sweep (default: 2 3 4)",
    )
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--n_seeds", type=int, default=4)
    parser.add_argument(
        "--datasets", type=int, nargs="*", default=None,
        help="Specific dataset IDs (default: auto-detect all)",
    )
    args = parser.parse_args()

    for opt in args.optimizers:
        if opt not in ALL_OPTIMIZERS:
            parser.error(f"Unknown optimizer '{opt}'. Choices: {ALL_OPTIMIZERS}")
    for sel in args.selections:
        if sel not in ALL_SELECTIONS:
            parser.error(f"Unknown selection '{sel}'. Choices: {ALL_SELECTIONS}")

    dataset_ids = args.datasets or discover_datasets()

    configs = []  # (optimizer, selection, pareto_tsize)
    for opt in args.optimizers:
        for sel in args.selections:
            if sel == "pareto":
                for k in args.pareto_tsizes:
                    configs.append((opt, sel, k))
            else:
                configs.append((opt, sel, 0))

    config_names = [
        f"{opt}_pareto_k{k}" if sel == "pareto" else f"{opt}_standard"
        for opt, sel, k in configs
    ]

    print(f"Datasets:       {dataset_ids}")
    print(f"Configs:        {config_names}")
    print(f"Pareto tsizes:  {args.pareto_tsizes}")
    print(f"Seeds:          0..{args.n_seeds - 1}")
    print(f"GPUs:           {args.n_gpus}")
    print()

    jobs = [
        (ds, seed, opt, sel, k)
        for opt, sel, k in configs
        for ds in dataset_ids
        for seed in range(args.n_seeds)
    ]
    total = len(jobs)
    print(f"Total jobs: {total}")
    print("=" * 50)

    t0 = time.time()
    run_batch(jobs, args.n_gpus)
    elapsed = time.time() - t0

    print("=" * 50)
    print(f"All done.  {total} jobs in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
