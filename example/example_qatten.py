from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Example of running StarCraft2 with RLlib QMIX.
This assumes all agents are homogeneous. The agents are grouped and assigned
to the multi-agent QMIX policy. Note that the default hyperparameters for
RLlib QMIX are different from pymarl's QMIX.
"""

import argparse
from gym.spaces import Tuple

import ray
from ray.tune import run_experiments, register_env

from value_mix import multiagent_smac_creator
from value_mix import QattenTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--map-name", type=str, default="3m")
    args = parser.parse_args()

    ray.init()
    register_env("sc2_grouped", multiagent_smac_creator)

    run_experiments(
        {
            "qatten_sc2_rmsprop": {
                "run": QattenTrainer,
                "env": "sc2_grouped",
                "stop": {
                    "training_iteration": args.num_iters,
                },
                "config": {
                    "mixing_embed_dim": 64,
                    "timesteps_per_iteration": 1000,
                    "target_network_update_freq": 500,
                    "buffer_size": 5000,
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
                        "epsilon_timesteps": 50000, #50000,  # Timesteps over which to anneal epsilon.
                    },
                    "evaluation_interval": 10,
                },
            },
        }
    )
