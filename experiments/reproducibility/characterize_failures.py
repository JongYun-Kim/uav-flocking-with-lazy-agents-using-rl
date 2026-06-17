"""Characterize the training-variance failure mode.

Not every seed yields a usable policy — and that is fine, because the failures are a single,
cheaply-detectable mode. From the validation screening (one row per checkpoint), for each
seed this reports: how close it ever gets to convergence; whether it has any converged
checkpoint (the selection rule needs one); and, for the seeds that never converge, the
signature of their lowest-total_L2 "trap" checkpoint — high non-convergence and collapsed
control effort (an all-lazy policy). Such a policy scores deceptively low on total_L2 (it
would beat the active baselines) yet never forms a flock, which is exactly why the
``nonconv_rate`` filter in ``select.py`` is mandatory and why these failures are caught
offline at zero deployment cost.

Usage: python -m experiments.reproducibility.characterize_failures \
    [--in experiments/reproducibility/results/screening.jsonl] [--nonconv 0.10]
"""
import argparse
import json
import os
import statistics as st
from collections import defaultdict

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_IN = os.path.join(WORKSPACE, "experiments", "reproducibility", "results", "screening.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--nonconv", type=float, default=0.10)
    args = ap.parse_args()

    by_seed = defaultdict(list)
    ref = None
    for line in open(args.inp):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("name") == "reference":
            ref = r
        if r.get("trial_seed") is not None:
            by_seed[r["trial_seed"]].append(r)

    if ref:
        print(f"reference: total_L2={ref['mean_total_L2']:.1f}  nonconv={ref['nonconv_rate']*100:.0f}%  "
              f"control_L2={ref['mean_ctrl_L2']:.1f}\n")

    hdr = (f"{'seed':>4} {'#ck':>4} {'minNC%':>7} | {'pass':>6} {'sel.L2':>7} | "
           f"{'trap.L2':>8} {'trap.NC%':>8} {'trap.ctrl':>9} | class")
    print(hdr)
    print("-" * len(hdr))
    converged, degenerate = [], []
    trap_nc, trap_ctrl, conv_ctrl = [], [], []
    for s in sorted(by_seed):
        rs = by_seed[s]
        min_nc = min(r["nonconv_rate"] for r in rs)
        passed = [r for r in rs if r["nonconv_rate"] <= args.nonconv]
        trap = min(rs, key=lambda r: r["mean_total_L2"])  # best total_L2 ignoring the filter
        if passed:
            sel = min(passed, key=lambda r: r["mean_total_L2"])
            converged.append(s)
            conv_ctrl.append(sel["mean_ctrl_L2"])
            sel_str, cls = f"{sel['mean_total_L2']:7.1f}", "converged"
            pf = f"{len(passed):>2}/{len(rs):<3}"
        else:
            degenerate.append(s)
            trap_nc.append(trap["nonconv_rate"])
            trap_ctrl.append(trap["mean_ctrl_L2"])
            sel_str, cls = f"{'--':>7}", "DEGENERATE"
            pf = f"{0:>2}/{len(rs):<3}"
        print(f"{s:>4} {len(rs):>4} {min_nc*100:>6.0f}% | {pf:>6} {sel_str} | "
              f"{trap['mean_total_L2']:>8.1f} {trap['nonconv_rate']*100:>7.0f}% "
              f"{trap['mean_ctrl_L2']:>9.1f} | {cls}")

    print(f"\n{len(converged)} converge {converged}\n{len(degenerate)} degenerate {degenerate}")
    if trap_nc:
        print("\n=== failure-mode signature (degenerate seeds' lowest-total_L2 checkpoint) ===")
        print(f"  non-convergence : mean {st.mean(trap_nc)*100:.0f}% "
              f"(range {min(trap_nc)*100:.0f}-{max(trap_nc)*100:.0f}%; filter cutoff {args.nonconv:.0%})")
        ctrl_ctx = f", reference {ref['mean_ctrl_L2']:.1f}" if ref else ""
        conv_ctx = f", converged-seeds {st.mean(conv_ctrl):.1f}" if conv_ctrl else ""
        print(f"  control effort  : mean control_L2 {st.mean(trap_ctrl):.1f} "
              f"(range {min(trap_ctrl):.1f}-{max(trap_ctrl):.1f}{ctrl_ctx}{conv_ctx})")
        print("  -> low control + high non-convergence = all-lazy collapse; deceptively low "
              "total_L2 but never converges, so the nonconv filter is mandatory.")


if __name__ == "__main__":
    main()
