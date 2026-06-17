"""Scalability sweep (8..1024 UAVs) for the reproducibility study.

For each selected checkpoint (and, optionally, the reference checkpoint and the ACS /
Heuristic baselines), evaluates total_L2 across swarm sizes. The trained 20-agent policy is
run at arbitrary sizes via the raw transformer ``policy_network`` — this reuses the exact
machinery of ``experiments/performance_benchmark/collect_scalability.py`` (imported below),
so the numbers are directly comparable to the paper's scalability figure.

Common random numbers: every method/size uses the same env seeds, so cross-method
differences are paired. Episodes use ``max_time_step=2000`` (large swarms need longer to
converge). Default ``--num-episodes 50`` matches the paper's scalability figure.

Results -> ``experiments/reproducibility/results/scalability/<label>_n<N>.json``
  (rl-s<seed>_n*, rl-ref_n*, acs_n*, heuristic_n*), ready for ``analyze_scalability.py``.

Usage:
    # every seed chosen by select.py, plus the reference checkpoint and baselines
    python -m experiments.reproducibility.scalability \
        --selected experiments/reproducibility/results/selected.json \
        --reference --baselines --device cuda --num-episodes 50
    # a single checkpoint
    python -m experiments.reproducibility.scalability --checkpoint <path> --label s2 --device cuda
"""
import argparse
import json
import os
import sys
import time

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

import ray

from experiments.performance_benchmark.collect_scalability import (
    MAX_TIME_STEP, collect_acs, collect_heuristic, collect_rl,
)
from experiments.performance_benchmark.config import (
    CHECKPOINT_PATH, SCALABILITY_AGENTS, SCALABILITY_SEEDS, build_env_config,
)

RESULTS_DIR = os.path.join(WORKSPACE, "experiments", "reproducibility", "results", "scalability")


def save_doc(out_dir, label, results, env_config):
    os.makedirs(out_dir, exist_ok=True)
    n = env_config["num_agents_max"]
    path = os.path.join(out_dir, f"{label}_n{n}.json")
    doc = {
        "method": label,
        "num_agents": n,
        "num_episodes": len(results),
        "seeds": sorted({r["seed"] for r in results}),
        "env_config": env_config,
        "results": sorted(results, key=lambda x: x["seed"]),
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"    saved {label}_n{n} ({len(results)} eps)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected", type=str, help="selected.json from select.py (run every seed in it)")
    ap.add_argument("--checkpoint", type=str, help="a single checkpoint to sweep")
    ap.add_argument("--label", type=str, default="checkpoint", help="label for a single --checkpoint")
    ap.add_argument("--reference", action="store_true",
                    help="also sweep the reference (featured) checkpoint as rl-ref")
    ap.add_argument("--baselines", action="store_true", help="also sweep ACS + Heuristic")
    ap.add_argument("--agents", type=int, nargs="*", default=SCALABILITY_AGENTS)
    ap.add_argument("--num-episodes", type=int, default=50)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--num-workers", type=int, default=16, help="Ray actors for RL inference")
    ap.add_argument("--num-cpus", type=int, default=48)
    ap.add_argument("--out-dir", type=str, default=RESULTS_DIR)
    args = ap.parse_args()

    # RL jobs: (label_without_rl_prefix, checkpoint_path)
    rl_jobs = []
    if args.selected:
        sel = json.load(open(args.selected))
        rl_jobs += [(f"s{e['seed']}", e["ckpt"]) for e in sel.get("seeds", [])]
    if args.checkpoint:
        rl_jobs.append((args.label, args.checkpoint))
    if args.reference:
        rl_jobs.append(("ref", CHECKPOINT_PATH))
    if not rl_jobs and not args.baselines:
        raise SystemExit("nothing to do: pass --selected / --checkpoint / --reference / --baselines")

    seeds = SCALABILITY_SEEDS[:args.num_episodes]
    print(f"[plan] sizes {args.agents} | n={len(seeds)} | "
          f"RL: {[lbl for lbl, _ in rl_jobs]} | baselines={args.baselines} | device={args.device}",
          flush=True)

    ray.init(num_cpus=args.num_cpus)
    t_total = time.time()
    for n in args.agents:
        print(f"\n[n={n}]", flush=True)
        cfg_rl = build_env_config(num_agents=n, max_time_step=MAX_TIME_STEP, use_heuristics=True)
        for label, ckpt in rl_jobs:
            t0 = time.time()
            results = collect_rl(cfg_rl, seeds, ckpt, args.device, args.num_workers)
            save_doc(args.out_dir, f"rl-{label}", results, cfg_rl)
            print(f"      rl-{label}: {time.time()-t0:.0f}s", flush=True)
        if args.baselines:
            save_doc(args.out_dir, "heuristic", collect_heuristic(cfg_rl, seeds), cfg_rl)
            cfg_acs = build_env_config(num_agents=n, max_time_step=MAX_TIME_STEP, use_heuristics=False)
            save_doc(args.out_dir, "acs", collect_acs(cfg_acs, seeds), cfg_acs)

    ray.shutdown()
    print(f"\nTotal wall-clock: {time.time()-t_total:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
