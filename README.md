# Energy-Efficient and Scalable UAV Flocking by Learning Lazy Behaviors Using Reinforcement Learning

This repository contains the training and evaluation code for the paper:

> J. Kim, M. Jung, H. Oh, A. Tsourdos, H.-S. Shin, "Energy-Efficient and Scalable UAV Flocking by Learning Lazy Behaviors Using Reinforcement Learning"

## Overview

A PPO agent learns a **laziness allocation policy** that selectively reduces control effort for individual UAVs in a flocking swarm, achieving energy-efficient convergence without sacrificing flocking performance. The policy is implemented as a Transformer encoder–decoder that takes per-agent state embeddings and outputs a per-agent laziness vector.

## Structure

```
train.py                          # PPO training entry point (Ray Tune)
env/envs.py                       # Gym environment (LazyAgentsCentralized)
models/
  lazy_allocator.py               # Transformer & MLP policy wrappers for RLlib
  transformer_modules/            # Encoder, decoder, pointer-net, attention layers
utils/
  metaheuristics.py               # SL-PSO optimizer (baseline)
  sl_pso_base.py                  # Standalone SL-PSO reference implementation
experiments/
  performance_benchmark/          # RL / ACS / PSO / heuristic evaluation scripts
  compute_benchmark/              # Inference speed & CUDA graph benchmarks
  analysis/                       # Paper figure & table generation
  mlp_ablation_study/             # MLP architecture ablation (train + eval)
docker/                           # Dockerfile and dependencies
```

## Requirements

See `docker/requirements.txt`. Key dependencies: Python 3.9+, PyTorch, Ray/RLlib, NumPy, Gym.

```sh
cd docker && bash build.sh        # build Docker image
```

## Training

```sh
python train.py
```

Launches a Ray Tune grid search over hyperparameters. Adjust `env_config` and model config dicts in `train.py` as needed.

## Evaluation

Evaluation scripts are in `experiments/performance_benchmark/`. Each writes results to `experiments/performance_benchmark/results/`.

```sh
python -m experiments.performance_benchmark.collect_rl          # trained RL policy
python -m experiments.performance_benchmark.collect_heuristic   # heuristic
python -m experiments.performance_benchmark.collect_pso         # SL-PSO baseline
python -m experiments.performance_benchmark.collect_acs         # fully-active (ACS)
python -m experiments.performance_benchmark.collect_scalability # scalability sweep
```

Trained checkpoints should be placed under `checkpoints/` (not included in the repository due to file size).

## Citation

```bibtex
@article{kim2026lazy,
  title   = {Energy-Efficient and Scalable UAV Flocking by Learning Lazy Behaviors Using Reinforcement Learning},
  author  = {Kim, Jongyun and Jung, Minjae and Oh, Hyondong and Tsourdos, Antonios and Shin, Hyo-Sang},
  year    = {2026}
}
```
