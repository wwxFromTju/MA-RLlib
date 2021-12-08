import numpy as np
from ray.rllib.utils.framework import try_import_torch

from value_mix.ma_controler import unroll_mac
from value_mix.mixers.vdn_mixer import VDNMixer
from value_mix.mixers.qmix_mixer import QMixer, QMixerV2
from value_mix.mixers.qplex_mixer import DMAQer, DMAQ_SI_Weight
from value_mix.mixers.qatten_mixer import QattenMixer

torch, nn = try_import_torch()

class OneHot:
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), torch.float32

class ValueMixLoss(nn.Module):
    def __init__(self,
                 model,
                 target_model,
                 mixer,
                 target_mixer,
                 n_agents,
                 n_actions,
                 double_q=True,
                 gamma=0.99):
        nn.Module.__init__(self)
        self.model = model
        self.target_model = target_model
        self.mixer = mixer
        self.target_mixer = target_mixer
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.double_q = double_q
        self.gamma = gamma

        self.onehot = OneHot(n_actions)

    def forward(self,
                rewards,
                actions,
                terminated,
                mask,
                obs,
                next_obs,
                action_mask,
                next_action_mask,
                state=None,
                next_state=None):
        """Forward pass of the loss.

        Args:
            rewards: Tensor of shape [B, T, n_agents]
            actions: Tensor of shape [B, T, n_agents]
            terminated: Tensor of shape [B, T, n_agents]
            mask: Tensor of shape [B, T, n_agents]
            obs: Tensor of shape [B, T, n_agents, obs_size]
            next_obs: Tensor of shape [B, T, n_agents, obs_size]
            action_mask: Tensor of shape [B, T, n_agents, n_actions]
            next_action_mask: Tensor of shape [B, T, n_agents, n_actions]
            state: Tensor of shape [B, T, state_dim] (optional)
            next_state: Tensor of shape [B, T, state_dim] (optional)
        """

        # Assert either none or both of state and next_state are given
        if state is None and next_state is None:
            state = obs  # default to state being all agents' observations
            next_state = next_obs
        elif (state is None) != (next_state is None):
            raise ValueError("Expected either neither or both of `state` and "
                             "`next_state` to be given. Got: "
                             "\n`state` = {}\n`next_state` = {}".format(
                                 state, next_state))

        # Calculate estimated Q-Values
        mac_out = unroll_mac(self.model, obs)

        # Pick the Q-Values for the actions taken -> [B * n_agents, T]
        chosen_action_qvals = torch.gather(
            mac_out, dim=3, index=actions.unsqueeze(3)).squeeze(3)

        x_mac_out = mac_out.clone().detach()
        x_mac_out[action_mask == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)
        

        # Calculate the Q-Values necessary for the target
        target_mac_out = unroll_mac(self.target_model, next_obs)

        # Mask out unavailable actions for the t+1 step
        ignore_action_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
        target_mac_out[ignore_action_tp1] = -np.inf

        # Max over target Q-Values
        if self.double_q:
            # Double Q learning computes the target Q values by selecting the
            # t+1 timestep action according to the "policy" neural network and
            # then estimating the Q-value of that action with the "target"
            # neural network

            # Compute the t+1 Q-values to be used in action selection
            # using next_obs
            mac_out_tp1 = unroll_mac(self.model, next_obs)

            # mask out unallowed actions
            mac_out_tp1[ignore_action_tp1] = -np.inf

            # obtain best actions at t+1 according to policy NN
            cur_max_actions = mac_out_tp1.argmax(dim=3, keepdim=True)

            # use the target network to estimate the Q-values of policy
            # network's selected actions
            target_max_qvals = torch.gather(target_mac_out, 3,
                                            cur_max_actions).squeeze(3)

            target_max_q_i = target_mac_out.max(dim=3)[0]
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        assert target_max_qvals.min().item() != -np.inf, \
            "target_max_qvals contains a masked action; \
            there may be a state with no valid actions."

        # Mix
        if not getattr(self, 'print_once', False):
            self.print_once = True
            print(f"self.mixer: {self.mixer}, \
                        self.mixer is VDNMixer: {isinstance(self.mixer, VDNMixer)}, \
                        self.mixer is QMixer: {isinstance(self.mixer, QMixer)}, \
                        self.mixer is QMixerV2: {isinstance(self.mixer, QMixerV2)}, \
                        self.mixer is QattenMixer: {isinstance(self.mixer, QattenMixer)}")
        if self.mixer is not None:
            if isinstance(self.mixer, QMixer) or isinstance(self.mixer, QMixerV2) or isinstance(self.mixer, QattenMixer):
                chosen_action_qvals = self.mixer(chosen_action_qvals, state)
                target_max_qvals = self.target_mixer(target_max_qvals, next_state)
            elif self.mixer is DMAQer:
                ans_chosen = self.mixer(chosen_action_qvals, state, is_v=True)
                ans_adv = self.mixer(chosen_action_qvals, state, actions=self.onehot(actions), max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv

                if self.double_q:
                    target_chosen = self.target_mixer(target_max_qvals, next_state, is_v=True)
                    target_adv = self.target_mixer(target_max_qvals, next_state, actions=self.onehot(cur_max_actions), max_q_i=target_max_q_i, is_v=False)

                    target_max_qvals = target_chosen + target_adv
                else:
                    raise "use double Q"

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error**2).sum() / mask.sum()
        return loss, mask, masked_td_error, chosen_action_qvals, targets