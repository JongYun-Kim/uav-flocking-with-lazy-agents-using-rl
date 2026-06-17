# Reproducibility & scalability-preservation study

This directory lets you reproduce, from scratch, the two claims behind the paper's
single-checkpoint results:

1. **Training reproducibility.** Training is stochastic, so we train an ensemble of
   independently-seeded runs and select each run's best checkpoint by a fixed, pre-registered
   rule (best validation `total_L2` subject to a convergence filter). Several independent
   seeds recover the reference (featured) checkpoint's 20-agent performance.
2. **Scalability preservation.** *Every* selected checkpoint keeps a flat control-cost curve
   as the swarm grows from 8 to 1024 UAVs, while the ACS and Heuristic baselines degrade — so
   the scalability property is intrinsic to the learned policy, not an artifact of one seed.

## Protocol

- **Metric:** `total_L2 = -reward_L2 = control_cost_L2 + convergence_time` (lower is better).
- **Training:** seeds 1–20, `w_control=0.018`, `w_vel=0.18` (fixed in `train.py`), 100 PPO
  iterations, every iteration checkpointed. Bitwise-reproducible (see `utils/seeding.py`).
- **Selection / reporting split (prevents selection bias):** checkpoints are *selected* on
  **validation** env seeds `5001..5100` and *reported* on the disjoint **test** seeds
  `1..1000`.
- **Convergence filter (mandatory):** a checkpoint is eligible only if `nonconv_rate ≤ 10%`.
  A degenerate all-lazy policy never forms a flock yet scores deceptively low on `total_L2`
  (convergence time caps out), so this filter is required — see `characterize_failures.py`.
- **Common random numbers:** every method/seed is evaluated on the same env seeds, so all
  comparisons are paired.

Run every command from the **repository root**. Evaluation reuses the shared engine in
`experiments/performance_benchmark/` (`config.py`, `collect_scalability.py`). Generated data
goes to `experiments/reproducibility/results/` (git-ignored).

## Pipeline

```sh
# 1. Train the ensemble (seeds 1..20). GPU-hours; checkpoints -> ~/ray_results/test_seed_control/
python -m experiments.reproducibility.train_seeds

# 2. Screen EVERY checkpoint on the validation seeds (+ reference checkpoint + ACS/Heuristic).
#    CPU-parallel; -> results/screening.jsonl
python -m experiments.reproducibility.evaluate \
    --results-dir ~/ray_results/test_seed_control --baselines

# 3. Select each seed's best converged checkpoint and classify vs the reference.
#    -> results/selected.json  (prints the per-seed table)
python -m experiments.reproducibility.select

# 4. (optional) Characterize the seeds that fail to train — one detectable non-convergence mode.
python -m experiments.reproducibility.characterize_failures

# 5. Re-evaluate the selected checkpoints on the disjoint TEST seeds 1..1000.
#    -> results/test.jsonl
python -m experiments.reproducibility.evaluate \
    --selected experiments/reproducibility/results/selected.json \
    --eval-seed-start 1 --n-eval 1000 \
    --out experiments/reproducibility/results/test.jsonl

# 6. Show that validation selection transfers to the unseen test seeds (anti-cherry-picking).
python -m experiments.reproducibility.val_test_transfer

# 7. Scalability sweep (8..1024) for the selected checkpoints + reference + baselines.
#    -> results/scalability/<label>_n<N>.json    (use --device cuda if you have GPUs)
python -m experiments.reproducibility.scalability \
    --selected experiments/reproducibility/results/selected.json \
    --reference --baselines --device cuda --num-episodes 50

# 8. Analyze: growth factors, each seed vs reference, CRN-paired CIs at N=1024.
python -m experiments.reproducibility.analyze_scalability
```

## What you should see

- Step 3: roughly a third of the 20 seeds reach the reference level; the rest are filtered out
  (degenerate / non-converging). The exact set is reproducible because training is seeded.
- Step 6: a small, stable validation→test gap and a high rank correlation — the selected
  checkpoints are genuinely good, not metric-overfit.
- Step 8: every selected checkpoint scales roughly flat (growth factor ≈ 1.0–1.4), while ACS
  and Heuristic roughly triple (≈ 2.7–3.4×).

## Scripts

| script | role |
|---|---|
| `train_seeds.py` | train seeds 1–20 (drives `train.py`) |
| `evaluate.py` | deterministic 20-agent evaluation — screen all checkpoints on validation seeds, or evaluate selected/single checkpoints on test seeds |
| `select.py` | apply the convergence filter, pick each seed's best checkpoint, classify vs the reference → `selected.json` |
| `characterize_failures.py` | show the training failures are one detectable non-convergence mode |
| `val_test_transfer.py` | validation→test generalization of the selection |
| `scalability.py` | scalability sweep (8→1024) for the selected checkpoints |
| `analyze_scalability.py` | growth factors, per-seed vs reference, paired CIs |

## Notes

- **Reference checkpoint:** defaults to the repo's featured checkpoint
  (`experiments/performance_benchmark/config.py::CHECKPOINT_PATH`); override with
  `--reference-checkpoint`.
- **Determinism:** `train.py` imports `utils.seeding`, which makes PPO training reproducible
  given `--seed` (avoid `--seed 0`: it is falsy and RLlib then skips per-env seeding).
- **Runtime:** training dominates (20 GPU runs). Screening is CPU-parallel over many cheap
  20-agent episodes. Scalability at N=1024 is the slow eval step (~minutes/size); use
  `--device cuda`.
