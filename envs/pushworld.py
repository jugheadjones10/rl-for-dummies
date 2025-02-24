import minigrid  # noqa
import jax.numpy as jnp
import flax.linen as nn
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper  # noqa
from pushworld.gym_env import PushWorldEnv
import os

from pushworld.config import BENCHMARK_PUZZLES_PATH


def pushworld_make_env(env_id, render_mode, **env_kwargs):
    env = PushWorldEnv(
        max_steps=50,
        puzzle_path=os.path.join(
            BENCHMARK_PUZZLES_PATH,
            "level0",
            "mini",
        ),
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
