from typing import Type

import flax.linen as nn
import jax.numpy as jnp


class Shallow1(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)
        return logits


class Shallow2(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)
        return logits


class Shallow3(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)
        return logits


# Roughly similar number of parameters to Shallow1
class Deep1(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(45)(x)
        x = nn.relu(x)
        x = nn.Dense(45)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)
        return logits


# Roughly similar number of parameters to Shallow2
class Deep2(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(65)(x)
        x = nn.relu(x)
        x = nn.Dense(65)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)
        return logits


# Roughly similar number of parameters to Shallow3
class Deep3(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(95)(x)
        x = nn.relu(x)
        x = nn.Dense(95)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)
        return logits


def get_network(network_name: str) -> Type[nn.Module]:
    if network_name not in globals():
        raise ValueError(
            f"Unknown network: {network_name}. Available networks: {[n for n in globals() if isinstance(globals()[n], type) and issubclass(globals()[n], nn.Module)]}"
        )
    return globals()[network_name]
