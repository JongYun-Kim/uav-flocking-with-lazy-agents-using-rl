# Energy-Efficient and Scalable UAV Flocking by Learning Lazy Behaviors Using Reinforcement Learning

[![DOI](https://zenodo.org/badge/1222387628.svg)](https://doi.org/10.5281/zenodo.20753941)

This repository contains the training and evaluation code for the paper:

> J. Kim, M. Jung, H. Oh, A. Tsourdos, H.-S. Shin, "Energy-Efficient and Scalable UAV Flocking by Learning Lazy Behaviors Using Reinforcement Learning"

## Overview

A PPO agent learns a **laziness allocation policy** that selectively reduces control effort for individual UAVs in a flocking swarm, achieving energy-efficient convergence without sacrificing flocking performance. The policy is implemented as a Transformer encoder–decoder that takes per-agent state embeddings and outputs a per-agent laziness vector.

## Structure

```
train.py                          # PPO training entry point (Ray Tune; --seed for reproducibility)
env/envs.py                       # Gym environment (LazyAgentsCentralized)
models/
  lazy_allocator.py               # Transformer & MLP policy wrappers for RLlib
  transformer_modules/            # Encoder, decoder, pointer-net, attention layers
utils/
  seeding.py                      # Seed control & RLlib determinism patch
  metaheuristics.py               # SL-PSO optimizer (baseline)
  sl_pso_base.py                  # Standalone SL-PSO reference implementation
experiments/
  performance_benchmark/          # RL / ACS / PSO / heuristic evaluation scripts
  compute_benchmark/              # Inference-latency benchmarks (speed, CUDA graphs, latency-vs-flock-size)
  reproducibility/                # Training-variance & scalability-preservation study
  analysis/                       # Paper figure & table generation
  mlp_ablation_study/             # MLP architecture ablation (train + eval)
checkpoints/transformer/          # Featured trained policy (RLlib format, tracked in-repo)
docker/                           # Dockerfile and dependencies
```

## Requirements

This is a **version-sensitive, old-stack RLlib** project; the recommended setup is the provided Docker image (CUDA 11.3 base). Exact pins live in `docker/requirements.txt`.

```sh
cd docker && bash build.sh        # build Docker image
```

Key pinned dependencies: Python 3.9, PyTorch 1.12.1 (cu113), Ray/RLlib 2.1.0, NumPy 1.23.4, classic Gym 0.23.1 (**not** Gymnasium).

## Training

```sh
python train.py                   # single reproducible run (default seed 42)
python train.py --seed 7          # reproducible run with a specific seed
python train.py --seeds 1 2 3 4   # one parallel Tune trial (and checkpoint) per seed
```

Launches a Ray Tune PPO run that trains the Transformer policy. `--seed` makes training reproducible (see **Reproducibility**); `--iters` sets the number of PPO iterations (default 100, checkpointed every iteration). Adjust `env_config` and the model config dicts in `train.py` as needed.

## Evaluation

Evaluation scripts are in `experiments/performance_benchmark/`. Each writes results to `experiments/performance_benchmark/results/`.

```sh
python -m experiments.performance_benchmark.collect_rl          # trained RL policy
python -m experiments.performance_benchmark.collect_heuristic   # heuristic
python -m experiments.performance_benchmark.collect_pso         # SL-PSO baseline
python -m experiments.performance_benchmark.collect_acs         # fully-active (ACS)
python -m experiments.performance_benchmark.collect_scalability # scalability sweep (8..1024 UAVs)
```

Evaluation loads the featured transformer checkpoint — included under `checkpoints/transformer/` (RLlib format) — via `CHECKPOINT_PATH` in `experiments/performance_benchmark/config.py`.

## Reproducibility

Training is seeded and bitwise-reproducible: `train.py --seed N` sets `config["seed"]` and, through `utils/seeding.py`, repairs RLlib 2.1.0's determinism setup *before* the policy is built, so the same seed reproduces the same model.

`experiments/reproducibility/` contains the full training-variance and scalability-preservation study: train an ensemble of independently-seeded runs, select each run's best checkpoint by a pre-registered rule on validation seeds, and report on disjoint test seeds. Every selected checkpoint keeps a flat control-cost curve from 8 to 1024 UAVs, showing the scalability property is intrinsic to the learned policy rather than an artifact of one seed. See `experiments/reproducibility/README.md` for the protocol and full pipeline.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{kim2026lazy,
  title   = {Energy-Efficient and Scalable UAV Flocking by Learning Lazy Behaviors Using Reinforcement Learning},
  author  = {Kim, Jongyun and Jung, Minjae and Oh, Hyondong and Tsourdos, Antonios and Shin, Hyo-Sang},
  year    = {2026}
}
```
