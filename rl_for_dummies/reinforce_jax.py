"""
REINFORCE in JAX/Flax.

This implementation closely follows the structure and logging style used in dqn_jax.py.
It defines a simple policy network, utilities for processing observations and computing
discounted returns, and a training loop that collects complete episodes before updates.
"""

import json
import os
from typing import Optional

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import random
import time
from dataclasses import dataclass

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

# Import special environment makers:
from envs.frozen_lake import frozen_lake_make_env
from envs.frozen_lake import get_network as frozen_lake_get_network
from envs.minigrid import minigrid_make_env
from envs.one import one_make_env

# from envs.pushworld import pushworld_make_env
from envs.shortcorridor import shortcorridor_make_env

# Print device information:
print("JAX devices: ", jax.devices())
print("Default device: ", jax.default_device())


@dataclass
class Args:
    env_kwargs: Optional[str] = None
    """JSON string containing additional kwargs for environments"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """the seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm-specific arguments:
    env_id: str = "FrozenLake-v1"
    """the id of the environment"""
    network: Optional[str] = None
    """Type of network to use"""
    total_timesteps: int = 100000
    """total timesteps of the experiment"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_envs: int = 1
    """the number of parallel game environments"""
    policy_visualisation_frequency: int = 1000
    """the frequency of policy visualisation"""


# Mapping of env-name prefixes to their special environment creators.
SPECIAL_ENVS = {
    "FrozenLake": (frozen_lake_make_env, frozen_lake_get_network),
    "MiniGrid": (minigrid_make_env, None),
    # "PushWorld": pushworld_make_env,
    "One": (one_make_env, None),
    "ShortCorridor": (shortcorridor_make_env, None),
}


def make_env(
    env_id,
    env_maker,
    seed,
    idx,
    capture_video,
    run_name,
    render_mode=None,
    **env_kwargs,
):
    def thunk():
        # Choose render mode: force "rgb_array" if recording
        use_render_mode = "rgb_array" if (capture_video and idx == 0) else render_mode

        # Create environment via SPECIAL_ENVS mapping or via gym.make
        if env_maker is None:
            env = gym.make(env_id, render_mode=use_render_mode, **env_kwargs)
        else:
            env = env_maker(env_id, use_render_mode, **env_kwargs)

        # Wrap in RecordVideo only if we're capturing video
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}", episode_trigger=lambda t: t % 100 == 0
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


class PolicyNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)

        return logits


def discount_returns(rewards, gamma):
    """Compute discounted returns for the provided rewards."""
    returns = np.zeros_like(rewards, dtype=np.float32)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Parse env_kwargs from JSON if provided
    env_kwargs = {}
    if args.env_kwargs:
        env_kwargs = json.loads(args.env_kwargs)

    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    for key, (env_make_fn, get_network_fn) in SPECIAL_ENVS.items():
        if args.env_id.startswith(key):
            env_maker = env_make_fn
            get_network = get_network_fn
            break

    if get_network is not None and args.network is not None:
        Network = get_network(args.network)
    else:
        Network = PolicyNetwork

    # TRY NOT TO MODIFY: seeding for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                env_maker,
                args.seed + i,
                i,
                args.capture_video,
                run_name,
                **env_kwargs,
            )
            for i in range(args.num_envs)
        ]
    )

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # Reset the environment and initialize policy network.
    obs, _ = envs.reset(seed=args.seed)

    # Initialize the policy network and create a TrainState
    policy = Network(action_dim=envs.single_action_space.n)
    state = TrainState.create(
        apply_fn=policy.apply,
        params=policy.init(init_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    # Optionally, you can jax.jit the apply_fn (this is safe because it's a pure function)
    policy.apply = jax.jit(policy.apply)

    # Refactored update function using TrainState
    @jax.jit
    def update(state, observations, actions, returns):
        def loss_fn(params):
            # Make a prediction for each observation (vectorized)
            logits = jax.vmap(lambda obs: policy.apply(params, obs))(observations)
            log_probs = jax.nn.log_softmax(logits)
            # Get the log probability associated with each action in the batch
            selected_log_probs = jnp.take_along_axis(
                log_probs, actions[:, None], axis=1
            ).squeeze()
            # Compute the REINFORCE loss (note the negative sign as we maximize log probabilities)
            loss = -jnp.mean(selected_log_probs * returns)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        # The apply_gradients method updates both parameters and the optimizer state
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    episode_observations = [[] for _ in range(args.num_envs)]
    episode_actions = [[] for _ in range(args.num_envs)]
    episode_rewards = [[] for _ in range(args.num_envs)]
    episode_count = 0
    episode_return = np.zeros(args.num_envs)

    start_time = time.time()

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # Compute action logits for each environment.
        # "obs" here has shape (num_envs, observation_dim)
        logits = policy.apply(state.params, obs)  # shape: (num_envs, num_actions)

        # Create a new set of subkeys – one for each environment.
        keys = jax.random.split(key, args.num_envs + 1)
        key = keys[0]  # update the main key
        subkeys = keys[1:]

        actions = jax.vmap(lambda sk, lg: jax.random.categorical(sk, lg))(
            subkeys, logits
        )
        actions = jax.device_get(actions)

        for i in range(args.num_envs):
            episode_observations[i].append(obs[i])
            episode_actions[i].append(int(actions[i]))

        next_obs, rewards, terminations, truncations, _ = envs.step(actions)

        for i in range(args.num_envs):
            episode_rewards[i].append(rewards[i])
            episode_return[i] += rewards[i]

        obs = next_obs

        for i in range(args.num_envs):
            if terminations[i] or truncations[i]:
                episode_count += 1

                disc_returns = discount_returns(
                    np.array(episode_rewards[i], dtype=np.float32), args.gamma
                )
                observations_arr = jnp.stack(episode_observations[i])
                actions_arr = jnp.array(episode_actions[i], dtype=jnp.int32)
                returns_arr = jnp.array(disc_returns, dtype=jnp.float32)

                state, loss = update(state, observations_arr, actions_arr, returns_arr)

                print(
                    f"global step: {global_step}, episode: {episode_count}, episodic_return={episode_return[i]:.2f}, episodic_length: {len(episode_rewards[i])}, policy_loss: {loss.item():.4f}"
                )
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                writer.add_scalar(
                    "charts/episodic_return", episode_return[i], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", len(episode_rewards[i]), global_step
                )
                writer.add_scalar("losses/policy_loss", loss.item(), global_step)

                # Reset the accumulators for this environment.
                episode_observations[i] = []
                episode_actions[i] = []
                episode_rewards[i] = []
                episode_return[i] = 0.0

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.reinforce_jax_model"
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(state.params))
        print(f"Model saved to {model_path}")

    writer.close()
    envs.close()
