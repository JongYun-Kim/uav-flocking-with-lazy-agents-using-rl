"""Select each seed's best checkpoint on the validation seeds, and classify it vs the reference.

Reads the screening JSONL written by ``evaluate.py`` (one row per checkpoint, evaluated on
the validation env seeds). Applies the MANDATORY convergence filter (``nonconv_rate <=
--nonconv``): a degenerate, all-lazy policy never converges yet scores deceptively low on
total_L2 (convergence time caps out), so it must be excluded. Among the converged checkpoints
of each seed it picks the lowest-total_L2 one, computes the common-random-number paired delta
vs the reference (featured) checkpoint, and reports how many seeds reach the reference level.

Writes ``selected.json`` (the chosen checkpoints) for ``scalability.py`` / ``evaluate.py
--selected``.

Usage:
    python -m experiments.reproducibility.select \
        [--in experiments/reproducibility/results/screening.jsonl] \
        [--nonconv 0.10] [--out experiments/reproducibility/results/selected.json]
"""
import argparse
import json
import os

import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(WORKSPACE, "experiments", "reproducibility", "results")
DEFAULT_IN = os.path.join(RESULTS_DIR, "screening.jsonl")
DEFAULT_OUT = os.path.join(RESULTS_DIR, "selected.json")


def load(path):
    rows = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[r["name"]] = r
    return rows


def paired(a, b):
    """CRN-paired mean delta a-b and 95% CI over the shared per-episode arrays."""
    a, b = np.array(a["ep_total_L2"]), np.array(b["ep_total_L2"])
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(n))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nonconv", type=float, default=0.10, help="max non-convergence rate (filter)")
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = load(args.inp)
    ref = rows.get("reference")
    heu, acs = rows.get("Heuristic"), rows.get("ACS")
    if ref is None:
        raise SystemExit("no 'reference' row in screening JSONL — run evaluate.py with "
                         "--reference-checkpoint (default: the featured checkpoint).")

    refs = [f"reference {ref['mean_total_L2']:.1f}"]
    if heu:
        refs.append(f"Heuristic {heu['mean_total_L2']:.1f}")
    if acs:
        refs.append(f"ACS {acs['mean_total_L2']:.1f}")
    print(f"validation-seed references: {' | '.join(refs)}")
    print(f"convergence filter: nonconv_rate <= {args.nonconv:.0%}\n")

    seeds = sorted({r["trial_seed"] for r in rows.values() if r.get("trial_seed") is not None})
    print(f"{'seed':>4} {'best@it':>8} {'L2':>7} {'nc%':>4} {'Δ vs reference (95% CI)':>24}  verdict")
    selected, equiv = [], []
    for s in seeds:
        cand = [r for r in rows.values()
                if r.get("trial_seed") == s and r["nonconv_rate"] <= args.nonconv]
        if not cand:
            print(f"{s:>4} {'--':>8} {'--':>7} {'--':>4} "
                  f"{'(no converged checkpoint - degenerate seed)':>24}")
            continue
        best = min(cand, key=lambda r: r["mean_total_L2"])
        dm, ci = paired(best, ref)
        verdict = ("BETTER than reference" if dm + ci < 0
                   else "~ reference" if dm - ci < 0 else "worse")
        if dm - ci < 0:
            equiv.append(s)
        print(f"{s:>4} {best['iter']:>8} {best['mean_total_L2']:>7.1f} "
              f"{best['nonconv_rate']*100:>4.0f} {dm:>+13.1f} ± {ci:>4.1f}      {verdict}")
        selected.append({
            "name": best["name"], "seed": s, "iter": best["iter"], "ckpt": best["ckpt"],
            "val_total_L2": best["mean_total_L2"], "nonconv_rate": best["nonconv_rate"],
            "delta_vs_reference": dm, "delta_ci95": ci, "verdict": verdict,
        })

    n_conv = len(selected)
    print(f"\nSUMMARY ({len(seeds)} seeds): {n_conv} converge; "
          f"{len(equiv)} reach the reference level (CI not worse): {equiv}")
    if equiv:
        eq_l2 = [r["val_total_L2"] for r in selected if r["seed"] in equiv]
        print(f"  among reference-level seeds: validation total_L2 "
              f"{np.mean(eq_l2):.1f} ± {np.std(eq_l2):.1f} (range {min(eq_l2):.1f}-{max(eq_l2):.1f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    doc = {
        "nonconv_threshold": args.nonconv,
        "reference": {"name": "reference", "ckpt": ref["ckpt"],
                      "val_total_L2": ref["mean_total_L2"], "nonconv_rate": ref["nonconv_rate"]},
        "n_seeds": len(seeds), "n_converged": n_conv,
        "reference_level_seeds": equiv,
        "seeds": sorted(selected, key=lambda r: r["seed"]),
    }
    json.dump(doc, open(args.out, "w"), indent=2)
    print(f"\nwrote {n_conv} selected checkpoints -> {args.out}")


if __name__ == "__main__":
    main()
