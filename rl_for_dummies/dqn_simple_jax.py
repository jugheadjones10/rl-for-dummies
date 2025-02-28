import json
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import random
import time
from dataclasses import dataclass
from typing import Optional

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
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


@dataclass
class Args:
    env_kwargs: Optional[str] = None
    """JSON string containing additional kwargs for environments"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
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

    # Algorithm specific arguments
    env_id: str = "FrozenLake-v1"
    """the id of the environment"""
    network: Optional[str] = None
    """Type of network to use"""
    total_timesteps: int = 5000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    start_e: float = 0.1
    """the starting epsilon for exploration"""
    end_e: float = 0
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.03
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_envs: int = 1
    """the number of parallel game environments"""


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


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)

    # env setup
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
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Get environment maker if it exists for this environment
    env_maker = None
    get_network = None
    for key, (env_make_fn, get_network_fn) in SPECIAL_ENVS.items():
        if args.env_id.startswith(key):
            env_maker = env_make_fn
            get_network = get_network_fn
            break

    if args.network is not None:
        # Network name will be in the form experiment-<experiment-name>-<network-name>
        if args.network.startswith("experiment-"):
            experiment_name, network_name = args.network[11:].split("-")
            try:
                # Import the get_network function from the experimental folder
                module_name = f"experiments.{experiment_name}.networks"
                network_module = __import__(module_name, fromlist=["get_network"])
                Network = network_module.get_network(network_name)
            except ImportError:
                raise ImportError(
                    f"Could not find experimental networks module: {module_name}"
                )
            except AttributeError:
                raise AttributeError(
                    f"Network '{network_name}' not found in experimental networks"
                )
        elif get_network_fn is not None:
            # Use environment-specific network getter
            Network = get_network_fn(args.network)
    else:
        Network = QNetwork

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                env_maker,
                args.seed + i,
                i,
                args.track,
                run_name,
                **env_kwargs,
            )
            for i in range(args.num_envs)
        ]
    )

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    obs, _ = envs.reset(seed=args.seed)
    q_network = Network(action_dim=envs.single_action_space.n)
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(init_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    q_network.apply = jax.jit(q_network.apply)

    @jax.jit
    def update(q_state, obs, actions, rewards, next_obs, dones):
        q_next = q_network.apply(q_state.params, next_obs)  # (batch_size, num_actions)
        q_next = jnp.max(q_next, axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * args.gamma * q_next  # (batch_size,)

        def mse_loss(params):
            q_pred = q_network.apply(params, obs)  # (batch_size, num_actions)
            q_pred = q_pred[
                jnp.arange(q_pred.shape[0]), actions.squeeze()
            ]  # (batch_size,)
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            q_state.params
        )
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state

    start_time = time.time()

    # Training loop
    obs, _ = envs.reset(seed=args.seed)
    dones = np.zeros(envs.num_envs)
    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = q_network.apply(q_state.params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)

        # Environment step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos:  # Changed from checking "final_info"
            print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            writer.add_scalar(
                "charts/episodic_return", infos["episode"]["r"], global_step
            )
            writer.add_scalar(
                "charts/episodic_length", infos["episode"]["l"], global_step
            )

        # This is a hack that only works for single environments. To support vectorized envs, we need to modify this.
        if dones[0] == 0:
            dones = np.logical_or(terminations, truncations)
            loss_value, old_val, q_state = update(
                q_state,
                obs,
                actions,
                rewards,
                next_obs,
                dones,
            )
        else:
            dones = np.zeros(envs.num_envs)

        # Logging
        if global_step % 100 == 0:
            writer.add_scalar("losses/td_loss", jax.device_get(loss_value), global_step)
            writer.add_scalar(
                "losses/q_values", jax.device_get(old_val).mean(), global_step
            )
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS",
                int(global_step / (time.time() - start_time)),
                global_step,
            )

        obs = next_obs

    envs.close()
    writer.close()
