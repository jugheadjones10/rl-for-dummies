import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import random

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

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


if __name__ == "__main__":
    # Setup
    SEED = 42
    learning_rate = 0.001
    gamma = 0.99
    epsilon = 0.1  # Fixed epsilon
    total_timesteps = 50000
    print_interval = 20

    # Seeding
    random.seed(SEED)
    np.random.seed(SEED)
    key = jax.random.PRNGKey(SEED)
    key, init_key = jax.random.split(key)

    # Environment
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env = OneHotWrapper(env)

    # Initialize network and optimizer
    q_network = QNetwork(action_dim=env.action_space.n)
    obs, _ = env.reset()
    params = q_network.init(init_key, obs)
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def get_q_values(params, obs):
        return q_network.apply(params, obs)

    @jax.jit
    def update(params, opt_state, obs, action, reward, next_obs, done):
        def loss_fn(params):
            q_values = q_network.apply(params, obs)
            current_q = q_values[action]
            next_q_values = q_network.apply(params, next_obs)
            next_q = jnp.max(next_q_values)
            target_q = reward + (1 - done) * gamma * next_q
            loss = 0.5 * (current_q - target_q) ** 2
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Training loop
    obs, _ = env.reset()
    score = 0.0
    episode_count = 0

    for step in range(total_timesteps):
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = get_q_values(params, obs)
            action = int(jnp.argmax(q_values))

        # Take step in environment
        next_obs, reward, done, truncated, _ = env.step(action)
        score += reward

        # Update Q-network
        params, opt_state, loss = update(
            params, opt_state, obs, action, reward, next_obs, done or truncated
        )

        obs = next_obs

        if done or truncated:
            obs, _ = env.reset()
            episode_count += 1

            if episode_count % print_interval == 0:
                print(
                    f"# of episode: {episode_count}, avg score: {score/print_interval}"
                )
                # Check if we've reached 95% success rate
                if score / print_interval == 0.95:
                    print(f"It took {episode_count} episodes to reach 95% score")
                    break
                score = 0.0

    env.close()
