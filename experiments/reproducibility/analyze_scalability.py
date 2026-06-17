"""Scalability analysis: is the flat-cost-curve property preserved across seeds?

Reads ``<results-dir>/{label}_n{N}.json`` (written by ``scalability.py``):
  * baselines ``acs`` / ``heuristic``,
  * the reference (featured) checkpoint ``rl-ref``,
  * one ``rl-s<seed>`` per selected checkpoint.

Reports per swarm size: mean total_L2; the N_min->N_max growth factor (≈1 = scales flat,
>>1 = degrades); each seed's delta vs the reference; and CRN-paired 95% CIs at the largest
size (each seed vs the reference and vs the better baseline). Common random numbers (same env
seeds per cell) make every delta paired.

Usage: python -m experiments.reproducibility.analyze_scalability [--results-dir DIR] [--nonconv-flag 1.5]
"""
import argparse
import glob
import json
import math
import os
import re

import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DIR = os.path.join(WORKSPACE, "experiments", "reproducibility", "results", "scalability")
T95 = 2.0096  # two-sided 95% t for df≈49 (n=50)


def pretty(m):
    if m in ("rl-ref", "rl-old"):
        return "RL ref"
    if m.startswith("rl-s"):
        return f"RL s{m[4:]}"
    return m.upper() if m == "acs" else m.capitalize()


def order_key(m):
    if m == "acs":
        return (0, 0)
    if m == "heuristic":
        return (1, 0)
    if m in ("rl-ref", "rl-old"):
        return (2, 0)
    if m.startswith("rl-s"):
        try:
            return (3, int(m[4:]))
        except ValueError:
            return (3, 9999)
    return (4, 0)


def is_seed(m):
    return m.startswith("rl-s")


def load(results_dir):
    data, ep = {}, {}  # data[m][N]=summary ; ep[m][N]={seed: total_L2} for pairing
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        mm = re.match(r"(.+)_n(\d+)\.json$", os.path.basename(f))
        if not mm:
            continue
        method, N = mm.group(1), int(mm.group(2))
        doc = json.load(open(f))
        rows = doc["results"]
        tl2 = np.array([-r["reward_L2"] for r in rows])
        mts = doc["env_config"]["max_time_step"]
        data.setdefault(method, {})[N] = dict(
            mean=float(tl2.mean()), std=float(tl2.std()), n=len(rows),
            nonconv=float(np.mean([r["episode_length"] >= mts for r in rows])),
        )
        ep.setdefault(method, {})[N] = {r["seed"]: -r["reward_L2"] for r in rows}
    return data, ep


def paired(a_by_seed, b_by_seed):
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    d = np.array([a_by_seed[s] - b_by_seed[s] for s in seeds], float)
    return d.mean(), T95 * d.std(ddof=1) / math.sqrt(len(d))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=DEFAULT_DIR)
    ap.add_argument("--degrade-flag", type=float, default=1.5,
                    help="growth factor above which a curve is flagged as degrading")
    args = ap.parse_args()

    data, ep = load(args.results_dir)
    if not data:
        print(f"no scalability results in {os.path.abspath(args.results_dir)}")
        return
    sizes = sorted({N for m in data.values() for N in m})
    methods = sorted(data, key=order_key)
    lo, hi = sizes[0], sizes[-1]
    ref = "rl-ref" if "rl-ref" in data else ("rl-old" if "rl-old" in data else None)
    print(f"results: {os.path.abspath(args.results_dir)}\nsizes: {sizes} | "
          f"n@{hi}: { {m: data[m][hi]['n'] for m in methods if hi in data[m]} }\n")

    print("=== mean total_L2 by swarm size (lower = better) ===")
    hdr = f"{'N':>6} | " + " | ".join(f"{pretty(m):>13}" for m in methods)
    print(hdr)
    print("-" * len(hdr))
    for N in sizes:
        cells = [(f"{data[m][N]['mean']:6.1f}±{data[m][N]['std']:4.0f}" if N in data[m] else "--")
                 for m in methods]
        print(f"{N:>6} | " + " | ".join(f"{c:>13}" for c in cells))

    print(f"\n=== growth factor total_L2(N={hi})/total_L2(N={lo}) (≈1 scales; >>1 degrades) ===")
    rank = []
    for m in methods:
        a, b = data[m].get(lo), data[m].get(hi)
        if a and b:
            fac = b["mean"] / a["mean"]
            flag = "  <-- DEGRADES" if fac >= args.degrade_flag else ""
            print(f"{pretty(m):>13}: {a['mean']:6.1f} -> {b['mean']:6.1f}  x{fac:.2f}  "
                  f"(nonconv@{hi} {b['nonconv']*100:.0f}%){flag}")
            if m.startswith("rl-"):
                rank.append((m, fac, b["mean"]))

    if ref and any(is_seed(m) for m in methods):
        print(f"\n=== each seed vs reference per size (Δ total_L2) ===")
        seeds = [m for m in methods if is_seed(m)]
        hdr = f"{'N':>6} | " + " | ".join(f"{pretty(m):>9}" for m in seeds)
        print(hdr)
        print("-" * len(hdr))
        for N in sizes:
            cells = []
            for m in seeds:
                if N in data[m] and N in data[ref]:
                    cells.append(f"{data[m][N]['mean']-data[ref][N]['mean']:+9.1f}")
                else:
                    cells.append(f"{'--':>9}")
            print(f"{N:>6} | " + " | ".join(cells))

    if rank:
        print(f"\n=== seed-sensitivity of scalability (sorted by growth factor) ===")
        for m, fac, hival in sorted(rank, key=lambda x: x[1]):
            flag = "" if fac < args.degrade_flag else "  <-- DEGRADES with scale"
            print(f"  {pretty(m):>9}: x{fac:.2f}  (total_L2@{hi}={hival:.1f}){flag}")

    # CRN-paired 95% CIs at the largest size: each seed vs reference and vs the better baseline.
    better_base = min((b for b in ("heuristic", "acs") if b in data and hi in data[b]),
                      key=lambda b: data[b][hi]["mean"], default=None)
    seeds = [m for m in methods if is_seed(m)]
    if seeds and hi in ep.get(ref or "", {}):
        print(f"\n=== CRN-paired deltas at N={hi} (95% CI) ===")
        if better_base:
            print(f"(reference total_L2@{hi}={data[ref][hi]['mean']:.1f}; "
                  f"better baseline = {pretty(better_base)} {data[better_base][hi]['mean']:.1f})")
        print(f"{'seed':>7} | {'Δ vs reference':>18} | {'vs ref?':>8} | {'Δ vs '+pretty(better_base) if better_base else '':>16}")
        for m in seeds:
            if hi not in ep[m]:
                continue
            dm, ci = paired(ep[m][hi], ep[ref][hi])
            verdict = "worse" if dm - ci > 0 else ("better" if dm + ci < 0 else "= ref")
            base_cell = ""
            if better_base and hi in ep[better_base]:
                bm, bci = paired(ep[m][hi], ep[better_base][hi])
                base_cell = f"{bm:+.1f} ± {bci:.1f}"
            print(f"{pretty(m)[3:]:>7} | {dm:>+11.1f} ± {ci:>4.1f} | {verdict:>8} | {base_cell:>16}")


if __name__ == "__main__":
    main()
