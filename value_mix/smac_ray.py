import random

import numpy as np
from gym.spaces import Discrete, Box, Dict, Tuple
from ray import rllib
from smac.env import StarCraft2Env

# for ppo/dqn/etc
def independent_smac_creator(smac_args):
    return RLlibStarCraft2Env(**smac_args)

# for vdn/qmix/qplex/ect
def multiagent_smac_creator(smac_args):
        env = RLlibStarCraft2Env(**smac_args)
        agent_list = list(range(env._env.n_agents))
        grouping = {
            "group_1": agent_list,
        }
        obs_space = Tuple([env.observation_space for i in agent_list])
        act_space = Tuple([env.action_space for i in agent_list])
        return env.with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space
        )


class RLlibStarCraft2Env(rllib.MultiAgentEnv):
    def __init__(self, **smac_args):
        self._env = StarCraft2Env(**smac_args)
        self._ready_agents = []
        self.observation_space = Dict(
            {
                "obs": Box(-1, 1, shape=(self._env.get_obs_size(),)),
                "action_mask": Box(0, 1, shape=(self._env.get_total_actions(),))
            }
        )
        self.action_space = Discrete(self._env.get_total_actions())

    def reset(self):
        obs_list, state_list = self._env.reset()
        return_obs = {}
        for i, obs in enumerate(obs_list):
            return_obs[i] = {
                "action_mask": np.array(self._env.get_avail_agent_actions(i)),
                # "avail_actions": np.array(self._env.get_avail_agent_actions(i)),
                "obs": obs,
            }

        self._ready_agents = list(range(len(obs_list)))
        return return_obs

    def step(self, action_dict):
        actions = []
        for i in self._ready_agents:
            if i not in action_dict:
                raise ValueError(
                    "You must supply an action for agent: {}".format(i)
                )
            actions.append(action_dict[i])

        if len(actions) != len(self._ready_agents):
            raise ValueError(
                "Unexpected number of actions: {}".format(
                    action_dict,
                )
            )

        rew, done, info = self._env.step(actions)
        obs_list = self._env.get_obs()
        return_obs = {}
        for i, obs in enumerate(obs_list):
            return_obs[i] = {
                "action_mask": self._env.get_avail_agent_actions(i),
                # "avail_actions": self._env.get_avail_agent_actions(i),
                "obs": obs,
            }
        rews = {i: rew / len(obs_list) for i in range(len(obs_list))}
        dones = {i: done for i in range(len(obs_list))}
        dones["__all__"] = done
        infos = {i: info for i in range(len(obs_list))}

        self._ready_agents = list(range(len(obs_list)))
        return return_obs, rews, dones, infos

    def close(self):
        """Close the environment"""
        self._env.close()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
