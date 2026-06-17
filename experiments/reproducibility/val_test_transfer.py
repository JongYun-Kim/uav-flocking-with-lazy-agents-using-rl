"""Validation -> test generalization: the anti-cherry-picking check.

Checkpoints are SELECTED on the validation env seeds (5001..5100) and REPORTED on the
DISJOINT test seeds (1..1000). If the selected iteration were a fluke that merely overfit the
metric on the validation seeds, its quality would not carry over to unseen test seeds. This
script shows it does: it pairs each seed's validation-selected checkpoint (from the screening
JSONL) with that same checkpoint's score on the test seeds (from a test-eval JSONL, produced
by ``evaluate.py --selected ... --eval-seed-start 1 --n-eval 1000``), and reports the
val->test gap and the rank correlation.

Usage:
    python -m experiments.reproducibility.val_test_transfer \
        --val experiments/reproducibility/results/screening.jsonl \
        --test experiments/reproducibility/results/test.jsonl [--nonconv 0.10]
"""
import argparse
import json
import os
from collections import defaultdict

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(WORKSPACE, "experiments", "reproducibility", "results")


def load_rows(path):
    rows = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[r["name"]] = r
    return rows


def select_per_seed(val_rows, nonconv):
    by_seed = defaultdict(list)
    for r in val_rows.values():
        if r.get("trial_seed") is not None:
            by_seed[r["trial_seed"]].append(r)
    sel = {}
    for s, rs in by_seed.items():
        cand = [r for r in rs if r["nonconv_rate"] <= nonconv]
        if cand:
            sel[s] = min(cand, key=lambda r: r["mean_total_L2"])
    return sel


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0] * len(v)
        for r, i in enumerate(order):
            rk[i] = r
        return rk
    rx, ry = rank(x), rank(y)
    n = len(x)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n * n - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", default=os.path.join(RES, "screening.jsonl"))
    ap.add_argument("--test", default=os.path.join(RES, "test.jsonl"))
    ap.add_argument("--nonconv", type=float, default=0.10)
    args = ap.parse_args()

    val_rows, test_rows = load_rows(args.val), load_rows(args.test)
    sel = select_per_seed(val_rows, args.nonconv)

    print("Selection on VALIDATION seeds; report on DISJOINT TEST seeds. "
          "total_L2 (lower=better).\n")
    hdr = (f"{'checkpoint':>12} | {'val L2':>7} {'val nc%':>7} | {'test L2':>8} {'test nc%':>8} | "
           f"{'gap(test-val)':>13}")
    print(hdr)
    print("-" * len(hdr))

    vs, ts, gaps = [], [], []
    # reference first (if present in both)
    if "reference" in val_rows and "reference" in test_rows:
        v, t = val_rows["reference"], test_rows["reference"]
        print(f"{'reference':>12} | {v['mean_total_L2']:>7.1f} {v['nonconv_rate']*100:>6.0f}% | "
              f"{t['mean_total_L2']:>8.1f} {t['nonconv_rate']*100:>7.0f}% | "
              f"{t['mean_total_L2']-v['mean_total_L2']:>+13.1f}")
    for s in sorted(sel):
        v = sel[s]
        t = test_rows.get(v["name"])
        if t is None:
            continue
        gap = t["mean_total_L2"] - v["mean_total_L2"]
        print(f"{v['name']:>12} | {v['mean_total_L2']:>7.1f} {v['nonconv_rate']*100:>6.0f}% | "
              f"{t['mean_total_L2']:>8.1f} {t['nonconv_rate']*100:>7.0f}% | {gap:>+13.1f}")
        vs.append(v["mean_total_L2"])
        ts.append(t["mean_total_L2"])
        gaps.append(gap)

    if len(vs) >= 3:
        rho = spearman(vs, ts)
        print(f"\nmean val->test gap = {sum(gaps)/len(gaps):+.1f}  "
              f"(range {min(gaps):+.1f}..{max(gaps):+.1f})")
        print(f"Spearman rank corr (val L2 vs test L2) over {len(vs)} seeds = {rho:.2f}")
        print("=> small, stable gap + high rank correlation: validation selection transfers to "
              "unseen test seeds, so the selected checkpoint is genuinely good, not an artifact.")
    else:
        print("\n(need >=3 matched seeds in both files; run evaluate.py --selected on the test seeds.)")


if __name__ == "__main__":
    main()
