import os
import time

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import random

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from torch.utils.tensorboard.writer import SummaryWriter

print("JAX devices: ", jax.devices())
print("Default device: ", jax.default_device())


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


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


learning_rate = 0.001
gamma = 0.99
epsilon = 0.1  # Fixed epsilon
SEED = 40


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def main():
    # Setup
    # Seeding
    random.seed(SEED)
    np.random.seed(SEED)
    key = jax.random.PRNGKey(SEED)
    key, init_key = jax.random.split(key)

    run_name = f"FrozenLake__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    # Environment
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env = OneHotWrapper(env)
    env.action_space.seed(SEED)

    # Initialize network and optimizer
    q_network = QNetwork(action_dim=env.action_space.n)

    s, _ = env.reset()
    params = q_network.init(init_key, s)
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    apply_fn = jax.jit(q_network.apply)

    @jax.jit
    def update(params, s, a, r, s_prime, done):
        q_next = apply_fn(params, s_prime)
        q_next = jnp.max(q_next, axis=-1)
        next_q_value = r + (1 - done) * gamma * q_next

        def loss_fn(params):
            q_values = apply_fn(params, s)
            current_q = q_values[a]
            loss = 0.5 * (current_q - next_q_value) ** 2
            return loss

        grads = jax.grad(loss_fn)(params)
        return grads

    time_taken = 0.0
    score = 0.0
    print_interval = 20

    for n_epi in range(5000):
        start_time = time.time()

        s, _ = env.reset()
        done = False

        returns = 0
        while not done:
            epsilon = linear_schedule(
                0.1,
                0,
                100,
                n_epi,
            )
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = apply_fn(params, s)
                action = int(jnp.argmax(q_values))
            s_prime, r, done, truncated, info = env.step(action)
            score += r
            returns += r
            # Train
            grads = update(params, s, action, r, s_prime, done or truncated)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            s = s_prime

        time_taken += time.time() - start_time

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        print(f"episode_number={n_epi}, episodic_return={returns}")
        writer.add_scalar("charts/episodic_return", returns, n_epi)
        returns = 0

        if n_epi % print_interval == 0 and n_epi != 0:
            print(
                f"# of episode: {n_epi}, avg score: {score/print_interval}, avg episode time: {time_taken/print_interval:.4f} sec"
            )
            score = 0.0
            time_taken = 0.0

    env.close()


if __name__ == "__main__":
    main()
