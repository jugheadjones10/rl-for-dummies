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

# Print device information:
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


class Policy(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)
        return logits


learning_rate = 0.0002
gamma = 0.98
SEED = 42


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    key = jax.random.PRNGKey(SEED)
    key, init_key = jax.random.split(key)

    env = gym.make("FrozenLake-v1", is_slippery=False)
    env = OneHotWrapper(env)
    pi = Policy(action_dim=env.action_space.n)

    s, _ = env.reset()
    params = pi.init(init_key, s)
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    apply_fn = jax.jit(pi.apply)

    @jax.jit
    def update(params, data):
        def loss_fn(params):
            R = 0.0
            loss = 0.0
            # Process episode in reverse to compute returns
            for s, a, r in reversed(data):
                R = r + gamma * R
                logits = apply_fn(params, s)
                log_prob = jax.nn.log_softmax(logits)[a]
                loss += log_prob * R
            return -loss / len(data)

        grads = jax.grad(loss_fn)(params)
        return grads

    time_taken = 0.0
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        start_time = time.time()

        s, _ = env.reset()
        done = False
        data = []

        while not done:
            logits = apply_fn(params, s)
            key, sample_key = jax.random.split(key)
            action = jax.random.categorical(sample_key, logits)
            action = int(action)
            s_prime, r, done, truncated, info = env.step(action)
            data.append((s, action, r))
            s = s_prime
            score += r

        grads = update(params, data)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        time_taken += time.time() - start_time

        if n_epi % print_interval == 0 and n_epi != 0:
            print(
                f"# of episode :{n_epi}, avg score : {score / print_interval}, avg episode time: {time_taken / print_interval:.4f} sec"
            )
            score = 0.0
            time_taken = 0.0

    env.close()


if __name__ == "__main__":
    main()
