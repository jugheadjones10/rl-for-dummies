from typing import Dict, Type

import flax.linen as nn
import gymnasium as gym
import jax.numpy as jnp
import numpy as np


class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.n = env.observation_space.n
        self.observation_space = gym.spaces.Box(0, 1, (self.n,), dtype=np.float32)

    def observation(self, obs):
        one_hot = np.zeros(self.n, dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot


# Will export the "make_env function?"
def frozen_lake_make_env(env_id, render_mode, **env_kwargs):
    env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
    env = OneHotWrapper(env)
    return env


class SimplePolicyNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)
        return logits


class DeepPolicyNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(8)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)
        return logits


# Dictionary mapping network names to their classes
NETWORKS: Dict[str, Type[nn.Module]] = {
    "simple": SimplePolicyNetwork,
    "deep": DeepPolicyNetwork,
}


def get_network(network_name: str) -> Type[nn.Module]:
    if network_name not in NETWORKS:
        raise ValueError(
            f"Unknown network: {network_name}. Available networks: {list(NETWORKS.keys())}"
        )
    return NETWORKS[network_name]
