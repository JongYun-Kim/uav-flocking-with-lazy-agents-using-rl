import argparse

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from env.envs import LazyAgentsCentralized, LazyAgentsCentralizedPendReward
from models.lazy_allocator import MyRLlibTorchWrapper, MyMLPModel
from utils.seeding import set_global_seed, enable_strict_determinism
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class MyCallbacks(DefaultCallbacks):
    def on_algorithm_init(self, *, algorithm, **kwargs):
        # The trial trains in a separate process (a Ray actor) from the launching driver, so the
        # driver's determinism setup does not reach it. The real fix is in
        # utils.seeding.patch_rllib_determinism(), installed on import of utils.seeding -- and
        # this callback's reference to enable_strict_determinism is what makes the actor import
        # utils.seeding before its Algorithm.setup() runs. The patch corrects RLlib's malformed
        # CUBLAS_WORKSPACE_CONFIG and enables use_deterministic_algorithms at the right moment
        # (before the policy's first GPU matmul). This call is an idempotent belt-and-suspenders
        # re-assertion inside the trainer process.
        enable_strict_determinism()

    def on_episode_start(self, worker, episode, **kwargs):
        episode.user_data["L1_reward_sum"] = 0
        episode.user_data["L2_reward_sum"] = 0

    def on_episode_step(self, worker, episode, **kwargs):
        from_infos = episode.last_info_for()["original_rewards"]
        episode.user_data["L1_reward_sum"] += from_infos[0]
        episode.user_data["L2_reward_sum"] += from_infos[1]

    def on_episode_end(self, worker, episode, **kwargs):
        episode.custom_metrics["episode_L1_reward_sum"] = episode.user_data["L1_reward_sum"]
        episode.custom_metrics["episode_L2_reward_sum"] = episode.user_data["L2_reward_sum"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train PPO (lazy_env) with seed control.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed for reproducibility (default: 42).")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="If given (e.g. --seeds 1 2 1 2), grid-search over these seeds as "
                             "parallel Tune trials -- one checkpoint per trial. Used for "
                             "reproducibility checks (same seed -> identical model). Overrides "
                             "--seed. Duplicates are allowed and produce distinct trials.")
    args = parser.parse_args()

    # One seed -> single reproducible run; many -> a grid_search of parallel trials.
    seeds = args.seeds if args.seeds else [args.seed]

    # Establish driver-side determinism BEFORE ray.init / any CUDA context creation (so
    # CUBLAS_WORKSPACE_CONFIG takes effect). RLlib also re-seeds each worker/env from
    # config["seed"] below; this call is the idempotent, auditable driver-side counterpart.
    set_global_seed(seeds[0])

    do_debug = False

    if do_debug:
        ray.init(local_mode=True)
        print("If not, you'd better set num_workers=0, num_gpu=0 for debugging purposes.")

    # register your custom environment
    num_agents_max = 20
    num_agents_min = 20
    env_config = {
        "num_agents_max": num_agents_max,  # Maximum number of agents
        "num_agents_min": num_agents_min,  # Minimum number of agents
        # Optional parameters
        "speed": 15,  # Speed in m/s. Default is 15
        "predefined_distance": 60,  # Predefined distance in meters. Default is 60
        # Tune the following parameters for your environment
        "std_pos_converged": 45,  # Standard position when converged. Default is 0.7*R
        "std_vel_converged": 0.1,  # Standard velocity when converged. Default is 0.1
        "std_pos_rate_converged": 0.1,  # Standard position rate when converged. Default is 0.1
        "std_vel_rate_converged": 0.2,  # Standard velocity rate when converged. Default is 0.2
        "max_time_step": 1000,  # Maximum time steps. Default is 2000,
        "incomplete_episode_penalty": -0,  # Penalty for incomplete episode. Default is -600
        "normalize_obs": True,  # If True, the env will normalize the obs. Default: False
        "use_fixed_horizon": True,  # If True, the env will use fixed horizon. Default: False
        "use_L2_norm": True,  # If True, the env will use L2 norm. Default: False
        # Step mode
        "auto_step": False,  # If True, the env will step automatically (i.e. episode length==1). Default: False
        # Ray config
        "use_custom_ray": False,  # If True, immutability of the env will be ensured. Default: False
        # For RL-lib models
        "use_preprocessed_obs": True,  # If True, the env will return preprocessed obs. Default: True
        "use_mlp_settings": False,  # If True, the env will use MLP settings. Default: False
    }
    env_name = "lazy_env"
    register_env(env_name, lambda cfg: LazyAgentsCentralizedPendReward(cfg))
    # Add keys "w_vel" and "w_control" to the env_config
    env_config["w_vel"] = 0.18
    env_config["w_control"] = 0.018

    # register your custom model
    custom_model_config_transformer = {
        "d_subobs": 5,
        "d_embed_input": 128,
        "d_embed_context": 128,
        "d_model": 128,
        "d_model_decoder": 128,
        "n_layers_encoder": 3,
        "n_layers_decoder": 1,
        "num_heads": 8,
        "d_ff": 512,
        "d_ff_decoder": 512,  # probably not used
        "clip_action_mean": 1.1,  # [0, clip_action_mean]
        "clip_action_log_std": 10.0,  # [-clip_action_log_std, -2]
        "dr_rate": 0,
        "norm_eps": 1e-5,
        "is_bias": False,
        "share_layers": True,
        "use_residual_in_decoder": True,
        "use_FNN_in_decoder": True,
        "use_deterministic_action_dist": True,
    }
    model_name_transformer = "custom_model"
    ModelCatalog.register_custom_model(model_name_transformer, MyRLlibTorchWrapper)

    custom_model_config_mlp = {
        "fc_sizes": [256, 256, 256],
        "fc_activation": "relu",
        "value_fc_sizes": [256, 256, 128],
        "value_fc_activation": "relu",
        "is_same_shape": False,  # avoid using this; let it be False unless you know what you are doing
        "share_layers": False,
    }
    model_name_mlp = "custom_model_mlp"
    ModelCatalog.register_custom_model(model_name_mlp, MyMLPModel)

    model_name_used = model_name_transformer
    if model_name_used == model_name_transformer:
        custom_model_config_used = custom_model_config_transformer
    elif model_name_used == model_name_mlp:
        custom_model_config_used = custom_model_config_mlp
        # Switch to MLP settings of the environment
        env_config["use_mlp_settings"] = True
    else:
        raise NotImplementedError("Unknown model name: {}".format(model_name_used))

    # train your custom model with PPO
    tune.run(
        "PPO",
        name="test_seed_control",
        stop={"training_iteration": 80},
        checkpoint_freq=1,
        keep_checkpoints_num=None,  # keep ALL checkpoints: every trial retains the same set of
                                    # iterations, so the verifier can compare the same iteration
                                    # across seeds without best-by-reward pruning removing it.
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        config={
            "env": env_name,
            "env_config": env_config,
            "framework": "torch",
            # --- Reproducibility: single seed from --seed (default 42). RLlib propagates
            # this to the driver, every rollout worker, and every env (via env.seed()). ---
            "seed": tune.grid_search(seeds) if len(seeds) > 1 else seeds[0],
            # Single --seed -> a scalar; --seeds 1 2 1 2 -> a grid_search of parallel trials.
            # (Avoid 0: it is falsy, so RLlib skips per-env env.seed() for seed 0.)
            "callbacks": MyCallbacks,
            "model": {
                "custom_model": model_name_used,
                "custom_model_config": custom_model_config_used,
            },
            "num_gpus": 0.5,
            "num_workers": 7,
            "num_envs_per_worker": 2,
            # Explicit to match the trained checkpoint: RLlib 2.1.0 defaults batch_mode to
            # "truncate_episodes", but the checkpoint was trained with "complete_episodes".
            "batch_mode": "complete_episodes",
            "rollout_fragment_length": 1000,
            "train_batch_size": 14000,
            "sgd_minibatch_size": 256,
            "num_sgd_iter": 36,
            "lr": 3e-5,
            # Must be fine-tuned when sharing vf-policy layers
            "vf_loss_coeff": 0.1,
            "use_critic": True,
            "use_gae": True,
            "gamma": 0.992,
            "lambda": 0.96,
            "kl_coeff": 0,  # no PPO penalty term; we use PPO-clip anyway; if none zero, be careful Nan in tensors!
            "clip_param": 0.25,
            "vf_clip_param": 20,
            "grad_clip": 40.0,
            "kl_target": 0.01,
        },
    )
