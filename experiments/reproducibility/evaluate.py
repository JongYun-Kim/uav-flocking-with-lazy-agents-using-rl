"""Evaluate trained laziness policies offline (deterministic, explore=False).

One engine, three sources of policies:

  * ``--results-dir DIR`` (SCREENING): evaluate EVERY checkpoint of EVERY seed-trial under a
    Ray results directory on the VALIDATION env seeds (default 5001..5100) -> one JSONL row
    per checkpoint. Also evaluates the reference (featured) checkpoint and, with
    ``--baselines``, the ACS / Heuristic baselines on the same seeds. Feeds ``select.py``.
  * ``--selected selected.json`` (FINAL / TEST): evaluate the checkpoints chosen by
    ``select.py`` on the disjoint TEST seeds (e.g. ``--eval-seed-start 1 --n-eval 1000``).
    Feeds ``val_test_transfer.py``.
  * ``--checkpoint PATH`` (single): evaluate one checkpoint.

Metric (paper, revision round 1): total_L2 = -reward_L2 = control_cost_L2 + convergence_time.
Mirrors ``experiments/performance_benchmark/collect_rl.py`` exactly (same env config,
explore=False, action clipped to [0, 1]). Episodes run on CPU over Ray actors (a 20-agent
episode is cheap; screening thousands of checkpoints is the workload). Output JSONL is
append-only and resumable (rows already present, keyed by ``name``, are skipped).

Examples:
    # 1) screen all checkpoints on the validation seeds (+ reference + baselines)
    python -m experiments.reproducibility.evaluate \
        --results-dir ~/ray_results/test_seed_control --baselines

    # 2) evaluate the selected checkpoints on the test seeds 1..1000
    python -m experiments.reproducibility.evaluate \
        --selected experiments/reproducibility/results/selected.json \
        --eval-seed-start 1 --n-eval 1000 \
        --out experiments/reproducibility/results/test.jsonl
"""
import argparse
import glob
import json
import os
import re
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # screening is CPU-only by design
os.environ.setdefault("OMP_NUM_THREADS", "1")

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

import numpy as np
import ray
from ray.util.actor_pool import ActorPool

from experiments.performance_benchmark.config import CHECKPOINT_PATH, VALIDATION_SEEDS

RESULTS_DIR = os.path.join(WORKSPACE, "experiments", "reproducibility", "results")


def run_episode(env, act_fn):
    """One deterministic episode; returns the benchmark's standard metrics dict."""
    from experiments.performance_benchmark.config import episode_metrics
    obs = env.reset()
    done, r1, r2 = False, 0.0, 0.0
    while not done:
        obs, _, done, info = env.step(act_fn(env, obs))
        r = info["original_rewards"]
        r1 += r[0]
        r2 += r[1]
    return episode_metrics(env, r1, r2, info)


def aggregate(name, trial_seed, it, ckpt, rows, eval_seeds, max_time_step, elapsed):
    tl2 = np.array([-r["reward_L2"] for r in rows])  # total cost (paper metric)
    tl1 = np.array([-r["reward_L1"] for r in rows])
    return {
        "name": name,
        "trial_seed": trial_seed,
        "iter": it,
        "ckpt": ckpt,
        "n": len(rows),
        "eval_seeds": [eval_seeds[0], eval_seeds[-1]],
        "mean_total_L2": float(tl2.mean()),
        "std_total_L2": float(tl2.std()),
        "mean_ctrl_L2": float(np.mean([r["control_cost_L2"] for r in rows])),
        "mean_conv_time": float(np.mean([r["convergence_time"] for r in rows])),
        "nonconv_rate": float(np.mean([r["episode_length"] >= max_time_step for r in rows])),
        "mean_total_L1": float(tl1.mean()),
        "elapsed_s": round(elapsed, 1),
        "ep_total_L2": [round(float(x), 4) for x in tl2],  # per-episode array for CRN pairing
        "ep_len": [int(r["episode_length"]) for r in rows],
    }


@ray.remote(num_cpus=1)
def baseline_task(kind, eval_seeds, num_agents, workspace):
    import sys
    import time
    import warnings
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    warnings.filterwarnings("ignore")
    from env.envs import LazyAgentsCentralized
    from experiments.performance_benchmark.config import build_env_config

    env_config = build_env_config(num_agents=num_agents, use_heuristics=(kind == "Heuristic"))
    t0 = time.time()
    rows = []
    for s in eval_seeds:
        env = LazyAgentsCentralized(env_config)
        env.seed(s)
        if kind == "ACS":
            cache = [None]

            def act_fn(env, obs, cache=cache):
                if cache[0] is None:
                    cache[0] = env.get_fully_active_action()
                return cache[0]
        else:
            def act_fn(env, obs):
                return env.compute_heuristic_action()
        rows.append(run_episode(env, act_fn))
    return aggregate(kind, None, None, None, rows, eval_seeds,
                     env_config["max_time_step"], time.time() - t0)


@ray.remote(num_cpus=1)
class Evaluator:
    def __init__(self, num_agents, workspace):
        import sys
        import warnings
        if workspace not in sys.path:
            sys.path.insert(0, workspace)
        warnings.filterwarnings("ignore")
        from ray.rllib.models import ModelCatalog
        from ray.tune.registry import register_env
        from env.envs import LazyAgentsCentralized
        from models.lazy_allocator import MyRLlibTorchWrapper
        ModelCatalog.register_custom_model("custom_model", MyRLlibTorchWrapper)
        register_env("lazy_env", lambda cfg: LazyAgentsCentralized(cfg))
        self.num_agents = num_agents

    def evaluate(self, task):
        import gc
        import time
        import warnings
        warnings.filterwarnings("ignore")
        import numpy as np
        from ray.rllib.policy.policy import Policy
        from env.envs import LazyAgentsCentralized
        from experiments.performance_benchmark.config import build_env_config

        name, trial_seed, it, ckpt, eval_seeds = task
        env_config = build_env_config(num_agents=self.num_agents, use_heuristics=True)
        t0 = time.time()
        policy = Policy.from_checkpoint(ckpt)
        policy.model.eval()

        def act_fn(env, obs):
            a = policy.compute_single_action(obs, explore=False)[0]
            return np.clip(a, 0.0, 1.0)

        rows = []
        for s in eval_seeds:
            env = LazyAgentsCentralized(env_config)
            env.seed(s)
            rows.append(run_episode(env, act_fn))

        out = aggregate(name, trial_seed, it, ckpt, rows, eval_seeds,
                        env_config["max_time_step"], time.time() - t0)
        del policy
        gc.collect()
        return out


def enumerate_checkpoints(results_dir, only_seeds=None, every=1,
                          iter_min=None, iter_max=None, max_ckpts=None):
    """List (name, seed, iter, policy_dir) for every checkpoint of every seed trial."""
    by_seed = {}
    for trial_dir in sorted(glob.glob(os.path.join(results_dir, "PPO_lazy_env_*"))):
        m = re.search(r"seed=(\d+)", trial_dir)
        if not m:
            continue
        seed = int(m.group(1))
        if only_seeds and seed not in only_seeds:
            continue
        for cdir in sorted(glob.glob(os.path.join(trial_dir, "checkpoint_*"))):
            it = int(cdir.rsplit("_", 1)[-1])
            if iter_min is not None and it < iter_min:
                continue
            if iter_max is not None and it > iter_max:
                continue
            pol = os.path.join(cdir, "policies", "default_policy")
            if os.path.isdir(pol):
                by_seed.setdefault(seed, []).append((it, pol))
    tasks = []
    for seed, items in by_seed.items():
        items.sort()
        keep = items[::every]
        if items[-1] not in keep:
            keep.append(items[-1])
        for it, pol in keep:
            tasks.append((f"s{seed}@{it}", seed, it, pol))
    tasks.sort(key=lambda t: (t[1], t[2]))
    return tasks[:max_ckpts] if max_ckpts else tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--results-dir", type=str,
                     help="screen ALL checkpoints under this Ray results dir (validation)")
    src.add_argument("--selected", type=str,
                     help="evaluate the checkpoints listed in select.py's selected.json")
    src.add_argument("--checkpoint", type=str, help="evaluate a single checkpoint")
    ap.add_argument("--label", type=str, default="checkpoint", help="label for a single --checkpoint")
    ap.add_argument("--num-agents", type=int, default=20)
    ap.add_argument("--eval-seed-start", type=int, default=VALIDATION_SEEDS[0])
    ap.add_argument("--n-eval", type=int, default=len(VALIDATION_SEEDS))
    ap.add_argument("--reference-checkpoint", type=str, default=CHECKPOINT_PATH,
                    help="reference (featured) checkpoint, added as a 'reference' row; '' to skip")
    ap.add_argument("--baselines", action="store_true",
                    help="also evaluate the ACS + Heuristic baselines on the same seeds")
    ap.add_argument("--num-actors", type=int, default=32)
    # screening sub-options
    ap.add_argument("--only-seeds", type=int, nargs="+", default=None)
    ap.add_argument("--every", type=int, default=1,
                    help="positional stride within each seed's iters (coarse pass; default 1=all)")
    ap.add_argument("--iter-min", type=int, default=None)
    ap.add_argument("--iter-max", type=int, default=None)
    ap.add_argument("--max-ckpts", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    eval_seeds = list(range(args.eval_seed_start, args.eval_seed_start + args.n_eval))
    default_name = ("screening.jsonl" if args.results_dir
                    else "test.jsonl" if args.selected else f"eval_{args.label}.jsonl")
    out = args.out or os.path.join(RESULTS_DIR, default_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # Build the policy task list + (optional) reference row.
    tasks = []
    if args.results_dir:
        tasks = [(n, s, i, p, eval_seeds) for n, s, i, p in
                 enumerate_checkpoints(os.path.expanduser(args.results_dir), args.only_seeds,
                                       args.every, args.iter_min, args.iter_max, args.max_ckpts)]
        if args.reference_checkpoint and os.path.isdir(args.reference_checkpoint):
            tasks.insert(0, ("reference", None, None, args.reference_checkpoint, eval_seeds))
    elif args.selected:
        sel = json.load(open(args.selected))
        for e in sel.get("seeds", []):
            tasks.append((e["name"], e.get("seed"), e.get("iter"), e["ckpt"], eval_seeds))
        ref = sel.get("reference", {}).get("ckpt") or (
            args.reference_checkpoint if os.path.isdir(args.reference_checkpoint) else None)
        if ref:
            tasks.insert(0, ("reference", None, None, ref, eval_seeds))
    else:
        tasks = [(args.label, None, None, args.checkpoint, eval_seeds)]

    # Resume: skip rows already written.
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                done.add(json.loads(line)["name"])
            except (json.JSONDecodeError, KeyError):
                pass
        if done:
            print(f"[resume] {len(done)} rows already in {out}", flush=True)
    tasks = [t for t in tasks if t[0] not in done]
    baselines = [k for k in (("ACS", "Heuristic") if args.baselines else ()) if k not in done]

    if not tasks and not baselines:
        print("[done] nothing to do")
        return
    print(f"[plan] {len(tasks)} policies + {len(baselines)} baselines | {args.num_agents} agents | "
          f"n={args.n_eval} seeds {eval_seeds[0]}..{eval_seeds[-1]} | {args.num_actors} actors",
          flush=True)

    ray.init(num_cpus=args.num_actors + 4, include_dashboard=False, log_to_driver=False)
    out_f = open(out, "a")

    def write(row):
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()

    t0 = time.time()
    base_futs = [baseline_task.remote(k, eval_seeds, args.num_agents, WORKSPACE) for k in baselines]
    n_written = 0
    if tasks:
        actors = [Evaluator.remote(args.num_agents, WORKSPACE)
                  for _ in range(min(args.num_actors, len(tasks)))]
        pool = ActorPool(actors)
        for row in pool.map_unordered(lambda a, t: a.evaluate.remote(t), tasks):
            n_written += 1
            write(row)
            rate = n_written / (time.time() - t0)
            eta = (len(tasks) - n_written) / rate / 60 if rate > 0 else float("inf")
            print(f"[{n_written}/{len(tasks)}] {row['name']:10s} "
                  f"L2 {row['mean_total_L2']:7.1f} nonconv {row['nonconv_rate']*100:3.0f}% "
                  f"({row['elapsed_s']:.0f}s) | ETA {eta:.0f} min", flush=True)
    for row in ray.get(base_futs):
        write(row)
        n_written += 1
        print(f"[baseline] {row['name']:9s} L2 {row['mean_total_L2']:7.1f} "
              f"nonconv {row['nonconv_rate']*100:3.0f}%", flush=True)

    out_f.close()
    print(f"[done] {n_written} rows in {(time.time()-t0)/60:.1f} min -> {out}", flush=True)
    ray.shutdown()


if __name__ == "__main__":
    main()
