import flax.linen as nn
import gymnasium as gym
import jax.numpy as jnp
import minigrid  # noqa
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper  # noqa


def minigrid_make_env(env_id, render_mode, **env_kwargs):
    env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    return env


class MiniGridQNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # Normalize the input
        x = x.astype(jnp.float32) / 255.0

        # Single conv layer since input is small (7x7)
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(1))(x)
        x = nn.relu(x)

        # Flatten the CNN output
        x = x.reshape((x.shape[0], -1))

        # Fully connected layers
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        # Output layer
        x = nn.Dense(self.action_dim)(x)
        return x
