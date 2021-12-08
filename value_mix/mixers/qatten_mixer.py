import numpy as np
import torch.nn.functional as F

from ray.rllib.utils.framework import try_import_torch

th, nn = try_import_torch()

class QattenMixer(nn.Module):
    def __init__(self, n_agents, state_shape):
        super(QattenMixer, self).__init__()

        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))

        # 4 + sc_env.shield_bits_ally + sc_env.unit_type_bits
        # self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        # unit_type_bits check smac map： m is 0
        self.agent_own_state_size = 4
        self.u_dim = int(np.prod(self.agent_own_state_size))

        self.n_query_embedding_layer1 =  64 # n_query_embedding_layer1
        self.n_query_embedding_layer2 = 32 # args.n_query_embedding_layer2
        self.n_key_embedding_layer1 = 32 # args.n_key_embedding_layer1
        self.n_head_embedding_layer1 = 64 # args.n_head_embedding_layer1
        self.n_head_embedding_layer2 = 4 # args.n_head_embedding_layer2
        self.n_attention_head = 4 # args.n_attention_head
        self.n_constrant_value = 32 # args.n_constrant_value

        self.query_embedding_layers = nn.ModuleList()
        for i in range(self.n_attention_head):
            self.query_embedding_layers.append(nn.Sequential(nn.Linear(self.state_dim, self.n_query_embedding_layer1),
                                                           nn.ReLU(),
                                                           nn.Linear(self.n_query_embedding_layer1, self.n_query_embedding_layer2)))
        
        self.key_embedding_layers = nn.ModuleList()
        for i in range(self.n_attention_head):
            self.key_embedding_layers.append(nn.Linear(self.u_dim, self.n_key_embedding_layer1))


        self.scaled_product_value = np.sqrt(self.n_query_embedding_layer2)

        self.head_embedding_layer = nn.Sequential(nn.Linear(self.state_dim, self.n_head_embedding_layer1),
                                                  nn.ReLU(),
                                                  nn.Linear(self.n_head_embedding_layer1, self.n_head_embedding_layer2))
        
        self.constrant_value_layer = nn.Sequential(nn.Linear(self.state_dim, self.n_constrant_value),
                                                  nn.ReLU(),
                                                  nn.Linear(self.n_constrant_value, 1))


    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        us = self._get_us(states)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        q_lambda_list = []
        for i in range(self.n_attention_head):
            state_embedding = self.query_embedding_layers[i](states)
            u_embedding = self.key_embedding_layers[i](us)

            # shape: [-1, 1, state_dim]
            state_embedding = state_embedding.reshape(-1, 1, self.n_query_embedding_layer2)
            # shape: [-1, state_dim, n_agent]
            u_embedding = u_embedding.reshape(-1, self.n_agents, self.n_key_embedding_layer1)
            u_embedding = u_embedding.permute(0, 2, 1)

            # shape: [-1, 1, n_agent]
            raw_lambda = th.matmul(state_embedding, u_embedding) / self.scaled_product_value
            q_lambda = F.softmax(raw_lambda, dim=-1)

            q_lambda_list.append(q_lambda)

        # shape: [-1, n_attention_head, n_agent]
        q_lambda_list = th.stack(q_lambda_list, dim=1).squeeze(-2)

        # shape: [-1, n_agent, n_attention_head]
        q_lambda_list = q_lambda_list.permute(0, 2, 1)

        # shape: [-1, 1, n_attention_head]
        q_h = th.matmul(agent_qs, q_lambda_list)

        # if self.args.type == 'weighted':
            # shape: [-1, n_attention_head, 1]
        w_h = th.abs(self.head_embedding_layer(states))
        w_h = w_h.reshape(-1, self.n_head_embedding_layer2, 1)

        # shape: [-1, 1]
        sum_q_h = th.matmul(q_h, w_h)
        sum_q_h = sum_q_h.reshape(-1, 1)
        # else:
        #     # shape: [-1, 1]
        #     sum_q_h = q_h.sum(-1)
        #     sum_q_h = sum_q_h.reshape(-1, 1)

        c = self.constrant_value_layer(states)
        q_tot = sum_q_h + c
        q_tot = q_tot.view(bs, -1, 1)
        return q_tot

    def _get_us(self, states):
        agent_own_state_size = self.agent_own_state_size
        with th.no_grad():
            us = states[:, :agent_own_state_size*self.n_agents].reshape(-1, agent_own_state_size)
        return us