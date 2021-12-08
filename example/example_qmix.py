import argparse
from gym.spaces import Tuple

import ray
from ray.tune import run_experiments, register_env

from value_mix import multiagent_smac_creator
from value_mix import QMixTrainer
from value_mix.configs import default_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--map-name", type=str, default="3m")
    parser.add_argument("--mixer-layer", type=str, default="v2")
    args = parser.parse_args()

    ray.init()
    register_env("sc2_grouped", multiagent_smac_creator)

    run_experiments(
        {
            "qmix_sc2_rmsprop": {
                "run": QMixTrainer,
                "env": "sc2_grouped",
                "stop": {
                    "training_iteration": args.num_iters,
                },
                "config": {
                    "mixer": "qmixv2" if args.mixer_layer == 'v2' else 'qmix',
                    "optimiser": "RMSprop", # "Adam"
                    "mixing_embed_dim": 64,
                    "timesteps_per_iteration": 100,
                    "target_network_update_freq": 500,
                    "buffer_size": 100,
                    "num_workers": args.num_workers,
                    "env_config": {
                        "map_name": args.map_name,
                    },
                     "exploration_config": {
                        # The Exploration class to use.
                        "type": "EpsilonGreedy",
                        # Config for the Exploration class' constructor:
                        "initial_epsilon": 1.0,
                        "final_epsilon": 0.02,
                        "epsilon_timesteps": 50000,  # Timesteps over which to anneal epsilon.
                    },
                },
            },
        }
    )
