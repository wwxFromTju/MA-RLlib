from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

def mac(model, obs, h):
    """Forward pass of the multi-agent controller.

    Args:
        model: TorchModelV2 class
        obs: Tensor of shape [B, n_agents, obs_size]
        h: List of tensors of shape [B, n_agents, h_size]

    Returns:
        q_vals: Tensor of shape [B, n_agents, n_actions]
        h: Tensor of shape [B, n_agents, h_size]
    """
    B, n_agents = obs.size(0), obs.size(1)
    if not isinstance(obs, dict):
        obs = {"obs": obs}
    obs_agents_as_batches = {k: _drop_agent_dim(v) for k, v in obs.items()}
    h_flat = [s.reshape([B * n_agents, -1]) for s in h]
    q_flat, h_flat = model(obs_agents_as_batches, h_flat, None)
    return q_flat.reshape(
        [B, n_agents, -1]), [s.reshape([B, n_agents, -1]) for s in h_flat]


def unroll_mac(model, obs_tensor):
    """Computes the estimated Q values for an entire trajectory batch"""
    B = obs_tensor.size(0)
    T = obs_tensor.size(1)
    n_agents = obs_tensor.size(2)

    mac_out = []
    h = [s.expand([B, n_agents, -1]) for s in model.get_initial_state()]
    for t in range(T):
        q, h = mac(model, obs_tensor[:, t], h)
        mac_out.append(q)
    mac_out = torch.stack(mac_out, dim=1)  # Concat over time

    return mac_out

def _drop_agent_dim(T):
    shape = list(T.shape)
    B, n_agents = shape[0], shape[1]
    return T.reshape([B * n_agents] + shape[2:])