import os
from typing import Type

import flax.linen as nn
import jax.numpy as jnp
from pushworld.config import BENCHMARK_PUZZLES_PATH
from pushworld.gym_env import PushWorldEnv


def pushworld_make_env(env_id, render_mode, **env_kwargs):
    env = PushWorldEnv(
        max_steps=50,
        puzzle_path=os.path.join(
            BENCHMARK_PUZZLES_PATH,
            "level0",
            "mini",
        ),
        render_mode=render_mode,
        **env_kwargs,
    )
    return env


# PushWorld DQN neural network architecture
class PushWorldQNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # 3 convolutional layers with ReLU activation
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(5, 5), strides=(1))(x)
        x = nn.relu(x)

        # Flatten the CNN output
        x = x.reshape((x.shape[0], -1))

        # 2 fully connected layers
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        # Output layer
        x = nn.Dense(self.action_dim)(x)
        return x


class SimplePushWorldQNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # 3 convolutional layers with ReLU activation
        x = nn.Conv(features=3, kernel_size=(3, 3), strides=(3))(x)
        x = nn.relu(x)

        # Flatten the CNN output
        x = x.reshape((x.shape[0], -1))

        # 2 fully connected layers
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        # Output layer
        x = nn.Dense(self.action_dim)(x)
        return x


def get_network(network_name: str) -> Type[nn.Module]:
    if network_name not in globals():
        raise ValueError(
            f"Unknown network: {network_name}. Available networks: {[n for n in globals() if isinstance(globals()[n], type) and issubclass(globals()[n], nn.Module)]}"
        )
    return globals()[network_name]
