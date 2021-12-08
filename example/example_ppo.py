import argparse

import ray
from ray.tune import run_experiments, register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from gym.spaces import Box

from value_mix import independent_smac_creator

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class ParametricActionsModel(DistributionalQTFModel):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(30, ),
                 action_embed_size=9,
                 **kw):
        print('this: ', obs_space)
        super(ParametricActionsModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)
        self.action_embed_model = FullyConnectedNetwork(
            Box(-1, 1, shape=true_obs_shape), action_space, action_embed_size,
            model_config, name + "_action_embed")

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]

        action_logits, _ = self.action_embed_model({
            "obs": input_dict["obs"]["obs"]
        })

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--map-name", type=str, default="3m")
    args = parser.parse_args()

    ray.init()

    register_env("smac", independent_smac_creator)
    ModelCatalog.register_custom_model("mask_model", ParametricActionsModel)

    run_experiments(
        {
            "ppo_sc2": {
                "run": "PPO",
                "env": "smac",
                "stop": {
                    "training_iteration": args.num_iters,
                },
                "config": {
                    "num_workers": args.num_workers,
                    "observation_filter": "NoFilter",  # breaks the action mask
                    "vf_share_layers": True,  # no separate value model
                    "framework": "tf",
                    "env_config": {
                        "map_name": args.map_name,
                    },
                    "model": {
                        "custom_model": "mask_model",
                    },
                },
            },
        }
    )
