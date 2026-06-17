"""Train the multi-seed ensemble for the reproducibility study.

Runs ``train.py`` once per random seed (default seeds 1..20). Each run is a fully
reproducible PPO training at the paper's reward coefficients (w_control=0.018, w_vel=0.18,
fixed in train.py) for ``--iters`` iterations. Checkpoints are written to the default Ray
results directory (``~/ray_results/test_seed_control/PPO_lazy_env_*_seed=<N>_*/``), with
every iteration kept, so the best checkpoint per seed is chosen afterwards by validation
(see ``evaluate.py`` + ``select.py``).

Runs are SEQUENTIAL (one trial at a time) — training 20 seeds x 100 iters is GPU-hours of
work. Train a subset with ``--seeds``, or call ``train.py --seeds 1 2 3 ...`` directly to
grid-search trials in parallel if you have the GPUs.

Usage:
    python -m experiments.reproducibility.train_seeds                  # seeds 1..20, 100 iters
    python -m experiments.reproducibility.train_seeds --seeds 1 2 3
    python -m experiments.reproducibility.train_seeds --iters 100
"""
import argparse
import os
import subprocess
import sys

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 21)),
                    help="random seeds to train (default: 1..20)")
    ap.add_argument("--iters", type=int, default=100,
                    help="PPO training iterations per seed (default: 100)")
    args = ap.parse_args()

    train_py = os.path.join(WORKSPACE, "train.py")
    seeds = [s for s in args.seeds if s != 0]  # train.py forbids seed 0 (falsy -> no seeding)
    if len(seeds) != len(args.seeds):
        print("note: seed 0 skipped (RLlib treats 0 as 'no seed').")
    print(f"Training {len(seeds)} seeds x {args.iters} iters (sequential): {seeds}\n")
    for i, seed in enumerate(seeds, 1):
        print(f"===== [{i}/{len(seeds)}] seed={seed} =====", flush=True)
        subprocess.run([sys.executable, train_py, "--seed", str(seed), "--iters", str(args.iters)],
                       cwd=WORKSPACE, check=True)

    print("\nAll training runs complete. Checkpoints under ~/ray_results/test_seed_control/.")
    print("Next: python -m experiments.reproducibility.evaluate "
          "--results-dir ~/ray_results/test_seed_control --baselines")


if __name__ == "__main__":
    main()
