import contextlib
import datetime
import os
import random
import signal
import sys
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper  # noqa
from pushworld import braindead  # noqa
from torch.distributions import Categorical
from torch.profiler import record_function
from torch.utils.tensorboard.writer import SummaryWriter

from envs.pushworld import PushWorldEnv


@dataclass
class TrainingConfig:
    """Configuration for Meta-A2C training on PushWorld environments"""

    # Environment settings
    puzzle_level: str = "level0"
    puzzle_category: str = "braindead"
    max_steps: int = 30

    # Training hyperparameters
    n_train_processes: int = 8
    learning_rate: float = 0.0002
    gamma: float = 0.98
    entropy_coef: float = 0.1

    # Meta-learning settings
    num_policy_updates: int = 2000
    meta_episode_length: int = 300
    meta_episodes_per_policy_update: int = 40
    # Number of meta episode collection iterations is meta_episodes_per_policy_update / n_train_processes

    # This should be a factor of meta_episodes_per_policy_update because we are reshaping all meta_episodes
    # into this batch size.
    meta_episodes_batch_size: int = 8
    # This means for each optimization epoch, we will need meta_episodes_per_policy_update / meta_episodes_batch_size
    # iterations to train on all the data.
    opt_epochs: int = 6

    # Checkpointing
    checkpoint: bool = False
    checkpoint_frequency: int = 10
    """how often to save model weights"""
    resume_from_checkpoint: Optional[str] = None

    # W&B
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Meta-RL PushWorld"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""

    # Output settings
    # output_dir: Path = Path("./results")
    # experiment_name: Optional[str] = None
    # print_interval: int = 5
    # enable_profiling: bool = False

    # Reproducibility
    seed: int = 42


def plot_results():
    # Create a figure with 2 subplots in 1 row
    plt.figure(figsize=(12, 5))

    # First subplot (left side)
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st position
    plt.plot(policy_update_mean_rewards)
    plt.title("Policy Update Mean Rewards")
    plt.xlabel("Update")
    plt.ylabel("Reward")

    # Second subplot (right side)
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd position
    plt.plot(avg_scores)
    plt.title("Test Mean Rewards")
    plt.xlabel("Update")
    plt.ylabel("Reward")

    # Add spacing between subplots
    plt.tight_layout()

    plt.savefig("training_progress.png")


def signal_handler(sig, frame):
    """Custom signal handler for graceful shutdown"""
    print("\nInterrupted by Ctrl+C. Cleaning up...")
    # Attempt to close environments safely
    if envs:
        envs.close()

    # Plot the results if we have any data
    if policy_update_mean_rewards and avg_scores:
        plot_results()

    print("Exiting gracefully...")
    sys.exit(0)  # Exit cleanly


signal.signal(signal.SIGINT, signal_handler)


def make_env(config, render_mode=None, puzzle_dir="train"):
    env = PushWorldEnv(
        max_steps=config.max_steps,
        puzzle_path=os.path.join(
            braindead.__path__[0],
            # config.puzzle_level,
            # config.puzzle_category,
            puzzle_dir,
        ),
        render_mode=render_mode,
        braindead=True,
        seed=config.seed,
    )
    return env


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(2, 4, kernel_size=2, stride=1)

        # Calculate output size after convolution:
        conv_output_size = 4 * 4 * 4

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.lstm_cell = nn.LSTMCell(conv_output_size + 6, 256)
        self.fc_pi = nn.Linear(256, 4)  # 4 actions for PushWorld
        self.fc_v = nn.Linear(256, 1)

    def initial_state(self, batch_size):
        return torch.zeros(batch_size, 2 * 256, dtype=torch.float)  # [B, 512]

    def forward(self, x, hidden, prev_action, prev_reward, prev_done):
        """Common forward pass for both policy and value networks"""
        # First scenario:
        # x: [B, 5, 5, 2]
        # prevs: [B]
        # hidden: [2, 256]

        # Second scenario:
        # x: [B, T, 5, 5, 2]
        # prevs: [B, T]
        # hidden: [2, 256] (this one is the same as the first scenario)

        # CNNs can only have one batch dimension
        # So if our state is of the form [B, T, H, W, C], we need to combine the B and T dimensions
        # first, and then re-separate them later.

        # NOTE: During testing, there is no batch dimensions. Should we just handle it there?
        # [H, W, C] (No batch dimension: this is during single env testing)
        # [B, T, H, W, C] (Batch and time dimensions: happens during training)

        # Possible prev_action/prev_reward/prev_done shapes:
        # [B] (No time dimension: this is during data collection)
        # [B, T] (Time dimension: happens during training)

        reshaped = False
        if len(x.shape) == 5:
            # For x: [B, T, H, W, C] -> [B*T, H, W, C]
            # For prevs: [B, T] -> [B*T]
            batch_size = x.shape[0]
            time_steps = x.shape[1]
            x = x.reshape(batch_size * time_steps, *x.shape[2:])
            prev_action = prev_action.reshape(batch_size * time_steps)
            prev_reward = prev_reward.reshape(batch_size * time_steps)
            prev_done = prev_done.reshape(batch_size * time_steps)
            reshaped = True

        # Convert from [B, H, W, C] to [B, C, H, W] for Conv2d
        x = x.permute(0, 3, 1, 2)

        # CNN feature extraction
        x = F.relu(self.conv1(x))

        # Flatten for fully connected layers
        x = x.reshape(x.size(0), -1)  # [B*T, conv_output_size]

        # Concat one-hot encoded prev_action, prev_reward, and prev_done to x
        prev_action_one_hot = F.one_hot(prev_action, num_classes=4)  # [B*T, 4]
        x = torch.cat(
            (
                x,
                prev_action_one_hot,
                prev_reward.unsqueeze(-1),
                prev_done.unsqueeze(-1),
            ),
            dim=-1,
        )  # [B*T, conv_output_size + 6]

        # Re-separate the batch and time dimensions
        if reshaped:
            # For x: [B*T, 64] -> [B, T, 64]
            # For prevs: [B*T] -> [B, T]
            x = x.reshape(batch_size, time_steps, -1)
            prev_action = prev_action.reshape(batch_size, time_steps)
            prev_reward = prev_reward.reshape(batch_size, time_steps)
            prev_done = prev_done.reshape(batch_size, time_steps)
        else:
            # If not reshaped, means we are in the first scenario
            # where x is [B, 64]
            # So we add a single time dimension
            # [B, 64] -> [B, 1, 64]
            x = x.unsqueeze(1)

        # Now we step through time steps, using LSTM Cell
        T = x.shape[1]
        features_by_timestep = []
        for t in range(T):
            h_n, c_n = hidden.chunk(2, dim=1)
            x_t = x[:, t, :]
            h_n, c_n = self.lstm_cell(x_t, (h_n, c_n))
            features_by_timestep.append(h_n)
            hidden = torch.cat((h_n, c_n), dim=1)

        features = torch.stack(features_by_timestep, dim=1)  # [B, T, 256]
        if T == 1:
            features = features.squeeze(1)  # [B, 256]

        # If T == 1, features is [B, 256]
        # Otherwise, features is [B, T, 256]
        # hidden is just [B, 512]
        return features, hidden

    def pi(self, x, hidden, prev_action, prev_reward, prev_done, softmax_dim=-1):
        """Policy network: returns action probabilities"""
        features, memory = self.forward(x, hidden, prev_action, prev_reward, prev_done)
        action_logits = self.fc_pi(features)
        prob = F.softmax(action_logits, dim=softmax_dim)
        return prob, memory

    def v(self, x, hidden, prev_action, prev_reward, prev_done):
        """Value network: returns state value estimate"""
        features, _ = self.forward(x, hidden, prev_action, prev_reward, prev_done)
        v = self.fc_v(features)  # [B, 1]
        return v


def worker(worker_id, master_end, worker_end, config):
    master_end.close()  # Forbid worker to use the master end for messaging
    env = make_env(config)
    env.action_space.seed(config.seed + worker_id)

    while True:
        cmd, data = worker_end.recv()
        if cmd == "step":
            ob, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            if done:
                ob, _ = env.reset(options={"maintain_puzzle": True})
            worker_end.send((ob, reward, done, info))
        elif cmd == "reset":
            ob, _ = env.reset()
            worker_end.send(ob)
        elif cmd == "close":
            worker_end.close()
            break
        elif cmd == "get_spaces":
            worker_end.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class ParallelEnv:
    def __init__(self, config):
        self.nenvs = config.n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(
            zip(master_ends, worker_ends)
        ):
            p = mp.Process(
                target=worker, args=(worker_id, master_end, worker_end, config)
            )
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Forbid master to use the worker end for messaging
        for worker_end in worker_ends:
            worker_end.close()

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(("reset", None))
        return np.stack([master_end.recv() for master_end in self.master_ends])

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):  # For clean up resources
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(("close", None))
        for worker in self.workers:
            worker.join()
            self.closed = True


def test(step_idx, model, config):
    env = make_env(config, puzzle_dir="test")
    env.action_space.seed(config.seed)
    done = False
    num_test = 100
    test_scores = []
    for _ in range(num_test):
        s, _ = env.reset()
        initial_state = model.initial_state(1)
        h_in = initial_state

        prev_action = torch.zeros(1, dtype=torch.long)  # [1]
        prev_reward = torch.zeros(1, dtype=torch.float)  # [1]
        prev_done = torch.ones(1, dtype=torch.float)  # [1] (not done)

        score = 0.0
        for t in range(config.meta_episode_length):
            # Convert observation to tensor and add batch dimension
            s_tensor = torch.from_numpy(s).float().unsqueeze(0)  # [1, H, W, C]

            prob, h_in = model.pi(
                s_tensor,
                h_in,
                prev_action,  # [1]
                prev_reward,  # [1]
                prev_done,  # [1]
            )

            a = Categorical(prob).sample().item()
            s_prime, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            if done:
                s_prime, _ = env.reset(options={"maintain_puzzle": True})

            s = s_prime
            score += r

            # Update prev_* values for next step
            prev_action = torch.tensor([a], dtype=torch.long)
            prev_reward = torch.tensor([r], dtype=torch.float)
            prev_done = torch.tensor([0.0 if done else 1.0], dtype=torch.float)

        test_scores.append(score)

    avg_score = np.mean(test_scores)
    print(f"Step # :{step_idx}, avg test score : {avg_score}")

    env.close()
    return avg_score


def compute_target(v_final, r_lst, mask_lst, config):
    G = v_final.squeeze(-1)
    td_target = list()

    # Note that in the meta-episodic setting of RL^2, the objective is
    # to maximize the expected discounted return of the meta-episode,
    # so we do not utilize the usual 'done' masking in this function.
    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        # G = r + gamma * G * mask
        G = r + config.gamma * G
        td_target.append(G)

    # Reverse td_target list
    return torch.tensor(td_target[::-1]).float()


def load_checkpoint(checkpoint_path, model, optimizer):
    """Load model and optimizer state from checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_update_idx = checkpoint["policy_update_idx"] + 1

    # Optionally restore tracking metrics
    if "policy_update_mean_rewards" in checkpoint:
        global policy_update_mean_rewards
        policy_update_mean_rewards = checkpoint["policy_update_mean_rewards"]

    if "avg_scores" in checkpoint:
        global avg_scores
        avg_scores = checkpoint["avg_scores"]

    return start_update_idx


# Data class object to store a meta episode data
@dataclass
class MetaEpisodeData:
    s_lst: np.ndarray
    a_lst: np.ndarray
    r_lst: np.ndarray
    done_lst: np.ndarray
    v_lst: np.ndarray
    adv_lst: np.ndarray


policy_update_mean_rewards = []
avg_scores = []
envs = None


def main(config: TrainingConfig, writer: SummaryWriter):
    # Seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    envs = ParallelEnv(config)
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    if config.resume_from_checkpoint:
        if config.track and config.resume_from_checkpoint.startswith("wandb:"):
            # Not sure if this works
            artifact_name = config.resume_from_checkpoint[6:]  # Remove "wandb:" prefix
            artifact = wandb.use_artifact(artifact_name)
            artifact_dir = artifact.download()
            checkpoint_path = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
        else:
            checkpoint_path = config.resume_from_checkpoint

        start_update_idx = load_checkpoint(checkpoint_path, model, optimizer)
    else:
        start_update_idx = 0

    # B is batch size, which in this case is n_train_processes
    # T is timesteps, which in this case is meta_episode_length
    for policy_update_idx in range(start_update_idx, config.num_policy_updates):
        meta_episodes = []
        for _ in range(
            config.meta_episodes_per_policy_update // config.n_train_processes
        ):
            s_lst, a_lst, r_lst, done_lst, v_lst = (
                list(),
                list(),
                list(),
                list(),
                list(),
            )

            # t means current timestep. tm1 is previous, tp1 is next.
            s_t = envs.reset()  # [B, 7, 7, 4]
            a_tm1 = np.zeros((config.n_train_processes,))  # [B]
            r_tm1 = np.zeros((config.n_train_processes,))  # [B]
            done_tm1 = np.zeros((config.n_train_processes,))  # [B]

            # Initial hidden state
            hidden_tm1 = model.initial_state(config.n_train_processes)  # [B, 512]

            for t in range(config.meta_episode_length):
                prob_t, hidden_t = model.pi(
                    torch.from_numpy(s_t).float(),
                    hidden_tm1,
                    torch.from_numpy(a_tm1).long(),
                    torch.from_numpy(r_tm1).float(),
                    torch.from_numpy(done_tm1).float(),
                )  # [B, 4], [B, 512]

                v_t = model.v(
                    torch.from_numpy(s_t).float(),
                    hidden_tm1,
                    torch.from_numpy(a_tm1).long(),
                    torch.from_numpy(r_tm1).float(),
                    torch.from_numpy(done_tm1).float(),
                )  # [B, 1]

                a_t = Categorical(prob_t).sample()  # [B]

                s_tp1, r_t, done_t, _ = envs.step(a_t.detach().numpy())

                # Store all t values
                s_lst.append(s_t)
                a_lst.append(a_t.detach().numpy())
                r_lst.append(r_t)
                done_lst.append(done_t)
                v_lst.append(v_t.squeeze(-1).detach().numpy())

                # Update values
                s_t = s_tp1
                a_tm1 = a_lst[-1]
                r_tm1 = r_lst[-1]
                done_tm1 = done_lst[-1]
                hidden_tm1 = hidden_t

            # Calculate td targets
            v_final = (
                model.v(
                    torch.from_numpy(s_t).float(),
                    hidden_tm1,
                    torch.from_numpy(a_tm1).long(),
                    torch.from_numpy(r_tm1).float(),
                    torch.from_numpy(done_tm1).float(),
                )
                .detach()
                .numpy()
            )  # [B, 1]
            td_target = compute_target(v_final, r_lst, done_lst, config)  # [T, B]

            v_vec = torch.tensor(v_lst)  # [T, B]
            # Both td_target and v_vec here are tensors. But do we need them to be?
            adv_lst = (td_target - v_vec).detach().numpy()  # [T, B]
            adv_lst = adv_lst.swapaxes(0, 1)  # [B, T]

            s_lst = np.stack(s_lst).swapaxes(0, 1)  # [B, T, 7, 7, 4]
            a_lst = np.stack(a_lst).swapaxes(0, 1)  # [B, T]
            r_lst = np.stack(r_lst).swapaxes(0, 1)  # [B, T]
            done_lst = np.stack(done_lst).swapaxes(0, 1)  # [B, T]
            v_lst = np.stack(v_lst).swapaxes(0, 1)  # [B, T]

            meta_episode_data = MetaEpisodeData(
                s_lst=s_lst,
                a_lst=a_lst,
                r_lst=r_lst,
                done_lst=done_lst,
                v_lst=v_lst,
                adv_lst=adv_lst,
            )
            meta_episodes.append(meta_episode_data)

        # Log mean rewards per meta episode
        mean_rewards = [round(ep.r_lst.sum(axis=1).mean(), 2) for ep in meta_episodes]
        print(f"Step #{policy_update_idx}, mean reward mean: {np.mean(mean_rewards)}")
        writer.add_scalar(
            "charts/mean_reward", np.mean(mean_rewards), policy_update_idx
        )
        policy_update_mean_rewards.append(np.mean(mean_rewards))

        # I want to "flatten" the meta episodes. Right now each meta_episode contains a batch of size n_train_processes.
        # Instead I want each meta episode to just contain a single episode of batch size 1.
        # Then during training we will randomly sample a batch of size B from the meta episodes for each policy update.
        flattened_meta_episodes = []
        for meta_episode in meta_episodes:
            for i in range(config.n_train_processes):
                flattened_meta_episodes.append(
                    MetaEpisodeData(
                        s_lst=meta_episode.s_lst[i],
                        a_lst=meta_episode.a_lst[i],
                        r_lst=meta_episode.r_lst[i],
                        done_lst=meta_episode.done_lst[i],
                        v_lst=meta_episode.v_lst[i],
                        adv_lst=meta_episode.adv_lst[i],
                    )
                )
        meta_episodes = flattened_meta_episodes

        # with profile(
        #     activities=[ProfilerActivity.CPU],
        # ) as prof:
        with contextlib.nullcontext() as prof:
            for opt_epoch in range(config.opt_epochs):
                with record_function("opt_epoch_loop"):
                    idxs = np.random.permutation(config.meta_episodes_per_policy_update)
                    total_loss = 0
                    for i in range(
                        0,
                        config.meta_episodes_per_policy_update,
                        config.meta_episodes_batch_size,
                    ):
                        idx_batch = idxs[i : i + config.meta_episodes_batch_size]
                        meta_episodes_batch = [meta_episodes[idx] for idx in idx_batch]

                        with record_function("data_prep"):
                            # Combine meta_episodes_batch into one MetaEpisodeData object that has a new batch dimension
                            # of size meta_episodes_batch_size.
                            meta_episode = MetaEpisodeData(
                                s_lst=np.stack(
                                    [ep.s_lst for ep in meta_episodes_batch]
                                ),
                                a_lst=np.stack(
                                    [ep.a_lst for ep in meta_episodes_batch]
                                ),
                                r_lst=np.stack(
                                    [ep.r_lst for ep in meta_episodes_batch]
                                ),
                                done_lst=np.stack(
                                    [ep.done_lst for ep in meta_episodes_batch]
                                ),
                                v_lst=np.stack(
                                    [ep.v_lst for ep in meta_episodes_batch]
                                ),
                                adv_lst=np.stack(
                                    [ep.adv_lst for ep in meta_episodes_batch]
                                ),
                            )

                            # Compute losses
                            mb_s = torch.from_numpy(
                                meta_episode.s_lst
                            ).float()  # [B, T, 7, 7, 4]
                            mb_a = torch.from_numpy(meta_episode.a_lst).long()  # [B, T]
                            mb_r = torch.from_numpy(
                                meta_episode.r_lst
                            ).float()  # [B, T]
                            mb_done = torch.from_numpy(
                                meta_episode.done_lst
                            ).float()  # [B, T]
                            mb_v = torch.from_numpy(
                                meta_episode.v_lst
                            ).float()  # [B, T]
                            mb_adv = torch.from_numpy(
                                meta_episode.adv_lst
                            ).float()  # [B, T]

                            a_dummy = torch.zeros(
                                (config.meta_episodes_batch_size, 1)
                            ).long()  # [B, 1]
                            r_dummy = torch.zeros(
                                (config.meta_episodes_batch_size, 1)
                            )  # [B, 1]
                            done_dummy = torch.zeros(
                                (config.meta_episodes_batch_size, 1)
                            )  # [B, 1]

                            prev_action = torch.cat(
                                (a_dummy, mb_a[:, :-1]), dim=1
                            )  # [B, T]
                            prev_reward = torch.cat(
                                (r_dummy, mb_r[:, :-1]), dim=1
                            )  # [B, T]
                            prev_done = torch.cat(
                                (done_dummy, mb_done[:, :-1]), dim=1
                            )  # [B, T]

                            hidden = model.initial_state(
                                config.meta_episodes_batch_size
                            )  # [B, 512]

                        with record_function("pi"):
                            prob, _ = model.pi(
                                mb_s,
                                hidden,
                                prev_action,
                                prev_reward,
                                prev_done,
                            )

                        with record_function("v"):
                            v = model.v(
                                mb_s,
                                hidden,
                                prev_action,
                                prev_reward,
                                prev_done,
                            )

                        with record_function("loss_computation"):
                            entropy = Categorical(prob).entropy()

                            # Use prob to calculate log prob again using mb_a
                            log_prob_a = torch.log(
                                prob.gather(2, mb_a.unsqueeze(-1)).squeeze(-1)
                            )

                            policy_loss = -torch.mean(log_prob_a * mb_adv)
                            value_loss = F.smooth_l1_loss(v.squeeze(-1), mb_v)
                            entropy_loss = -torch.mean(entropy)

                            loss = (
                                policy_loss
                                - config.entropy_coef * entropy_loss
                                + value_loss
                            )
                            total_loss += loss.item()

                    with record_function("optimizer_step"):
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

        # Print the profiler output (sorted by total CPU time)
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        # prof.export_chrome_trace("trace.json")

        avg_score = test(policy_update_idx, model, config)
        writer.add_scalar("charts/test_mean_reward", avg_score, policy_update_idx)
        avg_scores.append(avg_score)

        if config.checkpoint and policy_update_idx % config.checkpoint_frequency == 0:
            checkpoint_dir = f"checkpoints/{run_name}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_path = f"{checkpoint_dir}/checkpoint_{policy_update_idx}.pt"
            torch.save(
                {
                    "policy_update_idx": policy_update_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "policy_update_mean_rewards": policy_update_mean_rewards,
                    "avg_scores": avg_scores,
                },
                checkpoint_path,
            )

            if config.track:
                # Log as W&B Artifact
                artifact = wandb.Artifact(f"model-{run_name}", type="model")
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)


if __name__ == "__main__":
    # Parse command line arguments using tyro
    config = tyro.cli(TrainingConfig)

    run_name = f"{config.puzzle_level}__{config.puzzle_category}__{config.seed}__{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    if config.track:
        import wandb

        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            sync_tensorboard=True,
            config=vars(config),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )

    try:
        main(config, writer)
    finally:
        if envs:
            envs.close()
        plot_results()
