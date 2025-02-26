import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import random
import time

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from torch.utils.tensorboard.writer import SummaryWriter

# Import special environment makers:
from envs.frozen_lake import frozen_lake_make_env
from envs.frozen_lake import get_network as get_frozen_lake_network
from envs.minigrid import minigrid_make_env

# Mapping of env-name prefixes to their special env creators and network definitions.
SPECIAL_ENVS = {
    "FrozenLake": (frozen_lake_make_env, get_frozen_lake_network),
    "MiniGrid": (minigrid_make_env, None),
}


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


SEED = 42
if __name__ == "__main__":
    make_env, get_network = SPECIAL_ENVS["FrozenLake"]
    env = make_env("FrozenLake-v1", None, **{"is_slippery": False})
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.Autoreset(env)
    env.action_space.seed(SEED)

    run_name = f"FrozenLake__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    # Seeding
    random.seed(SEED)
    np.random.seed(SEED)
    key = jax.random.PRNGKey(SEED)
    key, init_key = jax.random.split(key)

    obs, _ = env.reset(seed=SEED)
    q_network = QNetwork(action_dim=env.action_space.n)
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(init_key, obs),
        tx=optax.adam(learning_rate=1e-3),
    )

    q_network.apply = jax.jit(q_network.apply)

    @jax.jit
    def update(q_state, obs, action, reward, next_obs, done):
        q_next = q_network.apply(q_state.params, next_obs)
        q_next = jnp.max(q_next, axis=-1)
        next_q_value = reward + (1 - done) * 0.99 * q_next

        def mse_loss(params):
            q_pred = q_network.apply(params, obs)
            q_pred = q_pred[action]
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            q_state.params
        )
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state

    start_time = time.time()

    # Training loop
    obs, _ = env.reset(seed=SEED)
    done = False
    for global_step in range(100000):
        if random.random() < 0.1:
            action = env.action_space.sample()
        else:
            q_values = q_network.apply(q_state.params, obs)
            action = q_values.argmax(axis=-1)
            action = int(jax.device_get(action))

        # Environment step
        next_obs, reward, terminations, truncations, infos = env.step(action)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos:  # Changed from checking "final_info"
            print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            writer.add_scalar(
                "charts/episodic_return", infos["episode"]["r"], global_step
            )
            writer.add_scalar(
                "charts/episodic_length", infos["episode"]["l"], global_step
            )

        if not done:
            done = terminations or truncations
            loss_value, old_val, q_state = update(
                q_state,
                obs,
                action,
                reward,
                next_obs,
                done,
            )
        else:
            done = False

        obs = next_obs

    env.close()
    writer.close()
