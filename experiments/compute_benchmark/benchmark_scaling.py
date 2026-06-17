"""
Inference-latency vs. flock-size sweep for the trained Transformer policy.

Answers the reviewer concern that Table 4 reports inference latency only at
``n = 20`` while the scalability study goes up to ``n = 1024``, and that the
encoder self-attention is ``O(n^2)``. This driver measures per-action-selection
wall-clock latency as a function of flock size for three GPU inference paths and
checks each against the environment's real-time control budget.

Variants measured (GPU only)
----------------------------
- ``RL``              -- full RLlib path: ``policy.compute_single_action`` (obs
                         preprocessing + batch build + forward + action-dist
                         sampling/clipping). The deployed "out-of-the-box" cost.
- ``RL_pure_forward`` -- raw ``policy.model.policy_network(obs_tensors)`` only,
                         obs pre-tensorised on-device outside the timer. This is
                         the "NN-only" cost and is exactly where the ``O(n^2)``
                         self-attention lives.
- ``cudagraph_fp32``  -- the same fp32 NN forward, captured + replayed with
                         ``torch.cuda.CUDAGraph`` (kernel-launch overhead removed).

Methodology matches ``benchmark.py`` / ``benchmark_cudagraph.py`` exactly (their
timing functions are imported and reused): per-policy warmup, then
``num_rollouts`` x ``steps_per_rollout`` timed samples with ``perf_counter_ns``
and ``torch.cuda.synchronize`` brackets; only the action-selection call is timed
(the subsequent ``env.step`` is executed so the obs distribution is
representative but not counted). The trained checkpoint (``num_agents_max=20``)
runs at arbitrary ``n`` because the model infers the sequence length from the obs
tensor shape (see ``models/lazy_allocator.py``).

Measurement integrity
---------------------
Everything runs **sequentially in a single process pinned to one GPU**. Latency
is a single-stream B=1 quantity; running sizes/variants concurrently (even on
separate GPUs) would contend on the CPU-side launch path (which dominates the
``RL`` variant) and corrupt the numbers. Pin the GPU from the shell, e.g.
``CUDA_VISIBLE_DEVICES=0``.

Real-time budget
----------------
The env advances by ``dt = 0.1 s`` per control step (``env/envs.py``), i.e. a
10 Hz control loop. A per-step inference latency below ``100 ms`` (=100000 us)
is therefore real-time. The budget is recorded in the output ``meta`` so the
analysis can report the latency/budget ratio at every scale.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python -m experiments.compute_benchmark.benchmark_scaling \
        --output experiments/compute_benchmark/results/scaling/latency_vs_n.json

    # smoke test (tiny sample counts, min + max size):
    CUDA_VISIBLE_DEVICES=0 python -m experiments.compute_benchmark.benchmark_scaling \
        --agents 20 1024 --num_rollouts 1 --steps_per_rollout 5 --warmup_steps 3 \
        --output /tmp/scaling_smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from typing import Dict, List, Optional

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Reuse the *exact* timing/setup code paths of the existing Table-4 benchmarks so
# the swept numbers are directly comparable to them.
from experiments.compute_benchmark.benchmark import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    build_env_config,
    collect_model_info,
    force_policy_device,
    time_policy,
    _read_first_cpu_model,
)
from experiments.compute_benchmark.benchmark_cudagraph import (  # noqa: E402
    _CUDAGraphForward,
    _copy_model,
    time_variant,
)

DEFAULT_AGENTS = [8, 16, 20, 32, 64, 128, 256, 512, 1024]
VARIANTS = ["RL", "RL_pure_forward", "cudagraph_fp32"]

# env control timestep (env/envs.py: self.dt default 0.1 s -> 10 Hz control)
DT_S = 0.1
REALTIME_BUDGET_US = DT_S * 1e6  # 100000.0


def adaptive_steps(n: int) -> int:
    """Timed samples per rollout, shrunk for large n.

    The forward latency stabilises in far fewer than 2000 samples, but the
    (untimed-yet-executed) ``env.step`` is ``O(n^2)`` in NumPy and dominates the
    wall-clock at large n. We keep the full 2000 (== Table 4) for n<=64 and step
    down beyond that. The realised sample count is always recorded per scale, so
    nothing is silently truncated.
    """
    if n <= 64:
        return 2000
    if n <= 256:
        return 1000
    if n <= 512:
        return 500
    return 300


def _strip_raw(summary: Dict[str, object], keep_raw: bool) -> Dict[str, object]:
    if not keep_raw:
        summary.pop("times_us", None)
    return summary


def measure_size(
    policy,
    n: int,
    *,
    num_rollouts: int,
    steps_per_rollout: int,
    warmup_steps: int,
    base_seed: int,
    max_time_step: int,
    keep_raw: bool,
) -> Dict[str, object]:
    """Run all three variants at one flock size, sequentially. Returns a dict
    keyed by variant name (each value is a timing summary or an ``error`` dict)."""

    import torch
    from env.envs import LazyAgentsCentralized

    env_config = build_env_config(n, max_time_step)
    env = LazyAgentsCentralized(env_config)

    out: Dict[str, object] = {
        "num_agents": n,
        "num_rollouts": num_rollouts,
        "steps_per_rollout": steps_per_rollout,
    }

    # ---- RL (full RLlib compute_single_action) ----
    try:
        s = time_policy(
            algo="RL", env=env, policy=policy,
            num_rollouts=num_rollouts, steps_per_rollout=steps_per_rollout,
            warmup_steps=warmup_steps, base_seed=base_seed, device="cuda",
        )
        out["RL"] = _strip_raw(s, keep_raw)
        print(f"    RL              mean={s['mean_us']:.1f}us  median={s['median_us']:.1f}us  "
              f"p95={s['p95_us']:.1f}us  p99={s['p99_us']:.1f}us  n={s['num_samples']}")
    except Exception as e:  # noqa: BLE001 - record + continue, never silently drop
        out["RL"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"    RL              FAILED: {type(e).__name__}: {e}")

    # ---- RL_pure_forward (raw NN forward) ----
    try:
        s = time_policy(
            algo="RL_pure_forward", env=env, policy=policy,
            num_rollouts=num_rollouts, steps_per_rollout=steps_per_rollout,
            warmup_steps=warmup_steps, base_seed=base_seed, device="cuda",
        )
        out["RL_pure_forward"] = _strip_raw(s, keep_raw)
        print(f"    RL_pure_forward mean={s['mean_us']:.1f}us  median={s['median_us']:.1f}us  "
              f"p95={s['p95_us']:.1f}us  p99={s['p99_us']:.1f}us  n={s['num_samples']}")
    except Exception as e:  # noqa: BLE001
        out["RL_pure_forward"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"    RL_pure_forward FAILED: {type(e).__name__}: {e}")

    # ---- cudagraph_fp32 (NN forward, CUDA-graph captured) ----
    forwarder = None
    try:
        net = _copy_model(policy, torch.float32)
        forwarder = _CUDAGraphForward(net, n, torch.float32, "cuda", name="cudagraph_fp32")
        s = time_variant(
            forwarder, env,
            num_rollouts=num_rollouts, steps_per_rollout=steps_per_rollout,
            warmup_steps=warmup_steps, base_seed=base_seed,
        )
        out["cudagraph_fp32"] = _strip_raw(s, keep_raw)
        print(f"    cudagraph_fp32  mean={s['mean_us']:.1f}us  median={s['median_us']:.1f}us  "
              f"p95={s['p95_us']:.1f}us  p99={s['p99_us']:.1f}us  n={s['num_samples']}")
    except Exception as e:  # noqa: BLE001
        out["cudagraph_fp32"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"    cudagraph_fp32  FAILED: {type(e).__name__}: {e}")
    finally:
        if forwarder is not None:
            del forwarder
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return out


def run(args: argparse.Namespace) -> Dict[str, object]:
    import torch
    from ray.rllib.models import ModelCatalog
    from ray.rllib.policy.policy import Policy
    from ray.tune.registry import register_env

    from env.envs import LazyAgentsCentralized
    from models.lazy_allocator import MyRLlibTorchWrapper

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required (this driver measures GPU latency only). "
            "Check CUDA_VISIBLE_DEVICES."
        )

    # Small B=1 forwards oversubscribe a big-core box with torch's default
    # thread count; pin it (matches benchmark.py / benchmark_cudagraph.py).
    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        try:
            torch.set_num_interop_threads(args.torch_threads)
        except RuntimeError:
            pass
    torch.backends.cudnn.benchmark = True

    ModelCatalog.register_custom_model("custom_model", MyRLlibTorchWrapper)
    register_env("lazy_env", lambda cfg: LazyAgentsCentralized(cfg))

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    print(f"Loading policy from {args.checkpoint}")
    policy = Policy.from_checkpoint(args.checkpoint)
    policy.model.eval()
    force_policy_device(policy, "cuda")
    print(f"Policy device: {next(policy.model.parameters()).device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    host_info = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "num_cpus": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    try:
        host_info["cpu_model"] = _read_first_cpu_model()
    except Exception:
        host_info["cpu_model"] = None

    # Param counts + weight/peak-inference memory (one-shot, restores device).
    model_info = collect_model_info(policy, 20)
    print(f"Model: total={model_info['total_params']:,} params "
          f"(policy_net={model_info['policy_network_params']:,})")

    agents = args.agents if args.agents else DEFAULT_AGENTS

    results: Dict[str, object] = {}
    for n in agents:
        steps = args.steps_per_rollout if args.steps_per_rollout else adaptive_steps(n)
        print(f"\n[n={n}]  rollouts={args.num_rollouts} x steps={steps} "
              f"(warmup={args.warmup_steps})")
        t0 = time.time()
        row = measure_size(
            policy, n,
            num_rollouts=args.num_rollouts,
            steps_per_rollout=steps,
            warmup_steps=args.warmup_steps,
            base_seed=args.base_seed,
            max_time_step=args.max_time_step,
            keep_raw=args.keep_raw,
        )
        row["wall_clock_s"] = time.time() - t0
        results[str(n)] = row
        print(f"  [n={n}] done in {row['wall_clock_s']:.1f}s")

    return {
        "meta": {
            "description": "Inference latency vs flock size for the trained "
                           "Transformer policy (reviewer revision).",
            "variants": VARIANTS,
            "agents": agents,
            "num_rollouts": args.num_rollouts,
            "steps_per_rollout": args.steps_per_rollout,  # None => adaptive
            "adaptive_steps": args.steps_per_rollout is None,
            "warmup_steps": args.warmup_steps,
            "base_seed": args.base_seed,
            "max_time_step": args.max_time_step,
            "device": "cuda",
            "dt_s": DT_S,
            "realtime_budget_us": REALTIME_BUDGET_US,
            "complexity_note": "Encoder self-attention is O(n^2). RL_pure_forward "
                               "and cudagraph_fp32 measure the NN forward where "
                               "that term lives; RL adds (n-independent) RLlib "
                               "preprocessing/sampling overhead on top.",
            "timing": "perf_counter_ns; cuda-synchronised; only the "
                      "action-selection call is timed (env.step executed but "
                      "not counted). Single process, single GPU, sequential.",
        },
        "host_info": host_info,
        "model_info": model_info,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, nargs="*", default=None,
                        help=f"flock sizes to sweep (default: {DEFAULT_AGENTS})")
    parser.add_argument("--num_rollouts", type=int, default=3)
    parser.add_argument("--steps_per_rollout", type=int, default=None,
                        help="fixed timed samples per rollout; default None => "
                             "adaptive_steps(n) (2000 for n<=64, down to 300 at 1024)")
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--base_seed", type=int, default=4242)
    parser.add_argument("--max_time_step", type=int, default=2000)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--keep_raw", action="store_true",
                        help="keep the per-sample times_us arrays in the output "
                             "(default: drop them, keep summary stats only)")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(THIS_DIR, "results", "scaling", "latency_vs_n.json"),
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    doc = run(args)
    with open(args.output, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
