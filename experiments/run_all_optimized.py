#!/usr/bin/env python
"""
Launch GP experiments for all datasets x all seeds across multiple GPUs.

Usage:
  python experiments/run_all.py                           # default: openes, 8 GPUs
  python experiments/run_all.py --optimizer shade          # single optimizer
  python experiments/run_all.py --optimizer all            # run every optimizer
  python experiments/run_all.py --n_gpus 4 --n_seeds 4    # fewer resources
"""
import argparse
import glob
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(PROJECT_ROOT, "experiments", "regressor_optimized.py")

# ALL_OPTIMIZERS = ["none", "cmaes", "openes", "snes", "xnes", "de", "shade"]
ALL_OPTIMIZERS = ["none", "openes", "xnes", "de", "shade"]


def discover_datasets():
    """Auto-detect dataset IDs from data/datasets/."""
    pattern = os.path.join(PROJECT_ROOT, "data", "datasets", "*_features.csv")
    ids = sorted(
        int(os.path.basename(p).split("_")[0])
        for p in glob.glob(pattern)
    )
    return ids


def run_batch(jobs, n_gpus):
    """Run a list of (dataset, seed, optimizer) jobs with at most n_gpus concurrent.

    Uses an explicit free-GPU pool so that no two processes share a GPU.
    """
    pending = list(jobs)
    free_gpus = list(range(n_gpus))
    running = {}  # gpu_id -> (proc, desc)

    while pending or running:
        # launch on every free GPU while there are pending jobs
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            dataset, seed, opt = pending.pop(0)
            cmd = [
                sys.executable, SCRIPT,
                "--dataset", str(dataset),
                "--seed", str(seed),
                "--gpu", str(gpu),
                "--optimizer", opt,
            ]
            desc = f"ds={dataset} seed={seed} opt={opt} gpu={gpu}"
            print(f"  LAUNCH  {desc}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            running[gpu] = (proc, desc)

        # poll for completion – return finished GPUs to the pool
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
    parser = argparse.ArgumentParser(description="Run all GP experiments")
    parser.add_argument(
        "--optimizer", type=str, default="all",
        help=f"Optimizer to use. 'all' runs every optimizer. Choices: all, {', '.join(ALL_OPTIMIZERS)}",
    )
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--n_seeds", type=int, default=4)
    parser.add_argument("--datasets", type=int, nargs="*", default=None,
                        help="Specific dataset IDs (default: auto-detect all)")
    args = parser.parse_args()

    dataset_ids = args.datasets or discover_datasets()
    optimizers = ALL_OPTIMIZERS if args.optimizer == "all" else [args.optimizer]

    print(f"Datasets:   {dataset_ids}")
    print(f"Optimizers: {optimizers}")
    print(f"Seeds:      0..{args.n_seeds - 1}")
    print(f"GPUs:       {args.n_gpus}")
    print()

    jobs = [
        (ds, seed, opt)
        for opt in optimizers
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
