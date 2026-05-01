import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from env.envs import LazyAgentsCentralized, LazyAgentsCentralizedPendReward
from models.lazy_allocator import MyRLlibTorchWrapper, MyMLPModel
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class MyCallbacks(DefaultCallbacks):
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

    do_debug = False

    if do_debug:
        ray.init(local_mode=True)
        print("If not, you'd better set num_workers=0, num_gpu=0 for debugging purposes.")

    # register your custom environment
    num_agents_max = 20
    num_agents_min = tune.grid_search([10, 20])
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
    env_config["w_vel"] = tune.grid_search([0.4, 0.2, 0.1, 0.05])
    env_config["w_control"] = tune.grid_search([0.001, 0.005, 0.01, 0.02])

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
        "share_layers": tune.grid_search([True, False]),
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
        name="expensive_tune_pend_rwd",
        stop={"training_iteration": 300},
        checkpoint_freq=1,
        keep_checkpoints_num=10,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        config={
            "env": env_name,
            "env_config": env_config,
            "framework": "torch",
            "callbacks": MyCallbacks,
            "model": {
                "custom_model": model_name_used,
                "custom_model_config": custom_model_config_used,
            },
            "num_gpus": 0.5,
            "num_workers": 7,
            "num_envs_per_worker": 3,
            "rollout_fragment_length": 1000,
            "train_batch_size": 21000,
            "sgd_minibatch_size": tune.grid_search([128, 512]),
            "num_sgd_iter": 38,
            "lr": tune.grid_search([1e-5, 4e-5]),
            # Must be fine-tuned when sharing vf-policy layers
            "vf_loss_coeff": 0.2,
            "use_critic": True,
            "use_gae": True,
            "gamma": tune.grid_search([0.99, 0.992, 0.995]),
            "lambda": tune.grid_search([0.9, 0.96, 0.98]),
            "kl_coeff": 0,  # no PPO penalty term; we use PPO-clip anyway; if none zero, be careful Nan in tensors!
            "clip_param": tune.grid_search([0.2, 0.24, 0.3]),
            "vf_clip_param": 20,
            "grad_clip": 40.0,
            "kl_target": 0.01,
        },
    )
