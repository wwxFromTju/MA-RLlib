import numpy as np
import torch.nn.functional as F

from ray.rllib.utils.framework import try_import_torch

th, nn = try_import_torch()

class DMAQ_SI_Weight(nn.Module):
    def __init__(self, n_agents, n_actions, state_dim, action_dim):
        super(DMAQ_SI_Weight, self).__init__()
  
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_action_dim = self.state_dim + self.action_dim

        self.num_kernel = 4 #args.num_kernel

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()

        adv_hypernet_embed = 64 #self.args.adv_hypernet_embed
        for i in range(self.num_kernel):  # multi-head attention
            self.key_extractors.append(nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed),
                                                        nn.ReLU(),
                                                        nn.Linear(adv_hypernet_embed, 1)))  # key
            self.agents_extractors.append(nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed),
                                                        nn.ReLU(),
                                                        nn.Linear(adv_hypernet_embed, self.n_agents)))  # agent
            self.action_extractors.append(nn.Sequential(nn.Linear(self.state_action_dim, adv_hypernet_embed),
                                                        nn.ReLU(),
                                                        nn.Linear(adv_hypernet_embed, self.n_agents)))  # action

    def forward(self, states, actions):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        data = th.cat([states, actions], dim=1)

        all_head_key = [k_ext(states) for k_ext in self.key_extractors]
        all_head_agents = [k_ext(states) for k_ext in self.agents_extractors]
        all_head_action = [sel_ext(data) for sel_ext in self.action_extractors]

        head_attend_weights = []
        for curr_head_key, curr_head_agents, curr_head_action in zip(all_head_key, all_head_agents, all_head_action):
            x_key = th.abs(curr_head_key).repeat(1, self.n_agents) + 1e-10
            x_agents = F.sigmoid(curr_head_agents)
            x_action = F.sigmoid(curr_head_action)
            weights = x_key * x_agents * x_action
            head_attend_weights.append(weights)

        head_attend = th.stack(head_attend_weights, dim=1)
        head_attend = head_attend.view(-1, self.num_kernel, self.n_agents)
        head_attend = th.sum(head_attend, dim=1)

        return head_attend


class DMAQer(nn.Module):
    def __init__(self, n_agents, state_shape, mixing_embed_dim, n_actions):
        super(DMAQer, self).__init__()

        self.n_agents = n_agents
        # self.embed_dim = mixing_embed_dim
        self.state_dim = int(np.prod(state_shape))

        self.n_actions = n_actions
        self.action_dim = n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1


        hypernet_embed =  mixing_embed_dim
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.n_agents))
        self.V = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                               nn.ReLU(),
                               nn.Linear(hypernet_embed, self.n_agents))

        self.si_weight = DMAQ_SI_Weight(n_agents, n_actions, self.state_dim, self.action_dim)

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = th.sum(agent_qs, dim=-1)
        return v_tot

    def calc_adv(self, agent_qs, states, actions, max_q_i):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        adv_w_final = self.si_weight(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)

        adv_tot = th.sum(adv_q * (adv_w_final - 1.), dim=1)
    
        return adv_tot

    def calc(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, states, actions, max_q_i)
            return adv_tot

    def forward(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)

        w_final = self.hyper_w_final(states)
        w_final = th.abs(w_final)
        w_final = w_final.view(-1, self.n_agents) + 1e-10
        v = self.V(states)
        v = v.view(-1, self.n_agents)

        agent_qs = w_final * agent_qs + v
       

        y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        return v_tot
