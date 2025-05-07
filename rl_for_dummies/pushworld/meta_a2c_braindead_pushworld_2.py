import datetime
import os
import random
import signal
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper  # noqa
from pushworld.data import braindead  # noqa
from pushworld.data.shuffle import shuffle_puzzles
from pushworld.gym_env import PushWorldEnv
from torch.distributions import Categorical
from torch.utils.tensorboard.writer import SummaryWriter


@dataclass
class TrainingConfig:
    """Configuration for Meta-A2C training on tabular MDP environments"""

    # Environment settings
    max_steps: int = 50

    # Training hyperparameters
    n_train_processes: int = 2
    learning_rate: float = 0.0002
    gamma: float = 0.99
    adam_weight_decay: float = 0.01
    adam_eps: float = 1e-5

    # Meta-learning settings
    num_policy_updates: int = 2000
    meta_episode_length: int = 500
    meta_episodes_per_policy_update: int = 32

    meta_episodes_batch_size: int = 8
    opt_epochs: int = 6

    # PPO params
    entropy_coef: float = 0.01
    clip_ratio: float = 0.10
    gae_lambda: float = 0.3

    # Braindead puzzles shuffle
    train_percentage: float = 0.01
    test_percentage: float = 0.01
    archive_percentage: float = 0.98

    # Checkpointing
    checkpoint: bool = False
    checkpoint_frequency: int = 200
    """how often to save model weights"""
    resume_from_checkpoint: Optional[str] = None

    # W&B
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Meta-RL PushWorld Braindead 2"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""

    # Reproducibility
    seed: int = 42


def signal_handler(sig, frame):
    """Custom signal handler for graceful shutdown"""
    print("\nInterrupted by Ctrl+C. Cleaning up...")
    print("Exiting gracefully...")
    sys.exit(0)  # Exit cleanly


signal.signal(signal.SIGINT, signal_handler)


def make_env(config, render_mode=None, puzzle_dir="train"):
    env = PushWorldEnv(
        max_steps=config.max_steps,
        puzzle_path=os.path.join(
            braindead.__path__[0],
            puzzle_dir,
        ),
        render_mode=render_mode,
        braindead=True,
        seed=config.seed,
    )
    return env


class RecurrentNetwork(nn.Module):
    def __init__(self, config):
        super(RecurrentNetwork, self).__init__()
        self.conv1 = nn.Conv2d(2, 4, kernel_size=2, stride=1)
        conv_output_size = 4 * 4 * 4
        self.lstm_cell = nn.LSTMCell(conv_output_size + 6, 256)
        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM weights using Xavier normal initialization"""
        # For each weight tensor in the LSTM cell
        for name, param in self.lstm_cell.named_parameters():
            if "weight" in name:
                # Apply Xavier normal initialization to all weight matrices
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                # Initialize bias terms to zero (LSTM has 2 bias vectors)
                nn.init.zeros_(param)

    def initial_state(self, batch_size):
        return torch.zeros(batch_size, 2 * 256, dtype=torch.float)  # [B, 512]

    def forward(self, x, hidden, prev_action, prev_reward, prev_done):
        """Common forward pass for both policy and value networks"""
        # First scenario:
        # x: [B, 7, 7, 4]
        # prevs: [B]
        # hidden: [2, 128]

        # Second scenario:
        # x: [B, T, 7, 7, 4]
        # prevs: [B, T]
        # hidden: [2, 128] (this one is the same as the first scenario)

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
            # For x: [B, T] -> [B*T]
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
        )  # [B*T, 294]

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
            # where x is [B, num_states+num_actions+2]
            # So we add a single time dimension
            # [B, num_states+num_actions+2] -> [B, 1, num_states+num_actions+2]
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


class PolicyNetwork(nn.Module):
    def __init__(self, config):
        super(PolicyNetwork, self).__init__()
        self.recurrent = RecurrentNetwork(config)
        self.fc_pi = nn.Linear(256, 4)

        nn.init.xavier_normal_(self.fc_pi.weight)
        nn.init.zeros_(self.fc_pi.bias)

    def initial_state(self, batch_size):
        return self.recurrent.initial_state(batch_size)

    def forward(self, x, hidden, prev_action, prev_reward, prev_done, softmax_dim=-1):
        features, hidden = self.recurrent(
            x, hidden, prev_action, prev_reward, prev_done
        )
        action_logits = self.fc_pi(features)
        return action_logits, hidden


class ValueNetwork(nn.Module):
    def __init__(self, config):
        super(ValueNetwork, self).__init__()
        self.recurrent = RecurrentNetwork(config)
        self.fc_v = nn.Linear(256, 1)

        nn.init.xavier_normal_(self.fc_v.weight)
        nn.init.zeros_(self.fc_v.bias)

    def initial_state(self, batch_size):
        return self.recurrent.initial_state(batch_size)

    def forward(self, x, hidden, prev_action, prev_reward, prev_done):
        features, hidden = self.recurrent(
            x, hidden, prev_action, prev_reward, prev_done
        )
        v = self.fc_v(features)
        return v, hidden


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


def test(step_idx, policy_net, config):
    env = make_env(config, puzzle_dir="test")
    env.action_space.seed(config.seed)
    done = False
    num_test = 5
    test_scores = []
    final_episode_scores = []
    for _ in range(num_test):
        s, _ = env.reset()
        initial_state = policy_net.initial_state(1)
        h_in = initial_state

        prev_action = torch.zeros(1, dtype=torch.long)  # [1]
        prev_reward = torch.zeros(1, dtype=torch.float)  # [1]
        prev_done = torch.ones(1, dtype=torch.float)  # [1] (not done)

        score = 0.0
        episode_score = 0.0
        episode_scores = []
        for t in range(config.meta_episode_length):
            # Convert observation to tensor and add batch dimension
            s_tensor = torch.from_numpy(s).float().unsqueeze(0)  # [1, H, W, C]

            prob, h_in = policy_net(
                s_tensor,
                h_in,
                prev_action,  # [1]
                prev_reward,  # [1]
                prev_done,  # [1]
            )

            a = Categorical(logits=prob).sample().item()
            s_prime, r, terminated, truncated, info = env.step(a)
            score += r
            episode_score += r
            done = terminated or truncated
            if done:
                episode_scores.append(episode_score)
                episode_score = 0.0
                s_prime, _ = env.reset(options={"maintain_puzzle": True})

            s = s_prime

            # Update prev_* values for next step
            prev_action = torch.tensor([a], dtype=torch.long)
            prev_reward = torch.tensor([r], dtype=torch.float)
            prev_done = torch.tensor([0.0 if done else 1.0], dtype=torch.float)

        if episode_scores:
            final_episode_scores.append(episode_scores[-1])
        else:
            final_episode_scores.append(0.0)

        test_scores.append(episode_score)

    avg_score = np.mean(test_scores)
    avg_final_episode_score = np.mean(final_episode_scores)
    print(
        f"Step # :{step_idx}, avg test score : {avg_score}, avg final episode score : {avg_final_episode_score}"
    )

    env.close()
    return avg_score, avg_final_episode_score


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


def compute_td_lambda_target(r_lst, v_lst, config):
    T = len(r_lst)
    td_lambda_target = np.zeros((T, config.n_train_processes), dtype=np.float32)
    adv_lst = np.zeros((T, config.n_train_processes), dtype=np.float32)
    for t in reversed(range(T)):
        r_t = r_lst[t]
        v_t = v_lst[t]
        v_tp1 = v_lst[t + 1] if t < T - 1 else 0.0
        adv_tp1 = adv_lst[t + 1] if t < T - 1 else 0.0
        delta_t = -v_t + r_t + config.gamma * v_tp1
        adv_t = delta_t + config.gamma * config.gae_lambda * adv_tp1
        adv_lst[t] = adv_t
        td_lambda_target[t] = v_t + adv_t

    return td_lambda_target, adv_lst


def load_checkpoint(
    checkpoint_path, policy_net, value_net, policy_optimizer, value_optimizer
):
    """Load model and optimizer state from checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    value_net.load_state_dict(checkpoint["value_net_state_dict"])
    policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
    value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
    start_update_idx = checkpoint["policy_update_idx"] + 1
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
    td_lambda_target: np.ndarray
    log_prob_a: np.ndarray


def get_weight_decay_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif len(param.shape) == 1:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def main(config: TrainingConfig, writer: SummaryWriter, envs: ParallelEnv):
    policy_net = PolicyNetwork(config)
    value_net = ValueNetwork(config)
    policy_optimizer = optim.AdamW(
        get_weight_decay_param_groups(policy_net, config.adam_weight_decay),
        lr=config.learning_rate,
        eps=config.adam_eps,
    )
    value_optimizer = optim.AdamW(
        get_weight_decay_param_groups(value_net, config.adam_weight_decay),
        lr=config.learning_rate,
        eps=config.adam_eps,
    )

    # B is batch size, which in this case is n_train_processes
    # T is timesteps, which in this case is meta_episode_length
    for policy_update_idx in range(0, config.num_policy_updates):
        # meta_episodes = []
        # Make a list of lists where first list is for each process, and second list contains meta episodes collected within each process
        meta_episodes = [[] for _ in range(config.n_train_processes)]
        with torch.no_grad():
            for _ in range(config.meta_episodes_per_policy_update):
                s_lst, a_lst, r_lst, done_lst, v_lst, log_prob_a_lst = (
                    list(),
                    list(),
                    list(),
                    list(),
                    list(),
                    list(),
                )

                # t means current timestep. tm1 is previous, tp1 is next.
                s_t = envs.reset()  # [B]
                a_tm1 = np.zeros((config.n_train_processes,))  # [B]
                r_tm1 = np.zeros((config.n_train_processes,))  # [B]
                done_tm1 = np.ones((config.n_train_processes,))  # [B]

                # Initial hidden state
                hidden_tm1_policy = policy_net.initial_state(
                    config.n_train_processes
                )  # [B, 512]
                hidden_tm1_value = value_net.initial_state(
                    config.n_train_processes
                )  # [B, 512]

                for t in range(config.meta_episode_length):
                    pi_dist_t, hidden_t_policy = policy_net(
                        torch.from_numpy(s_t).float(),
                        hidden_tm1_policy,
                        torch.from_numpy(a_tm1).long(),
                        torch.from_numpy(r_tm1).float(),
                        torch.from_numpy(done_tm1).float(),
                    )  # [B, num_actions], [B, 512]

                    v_t, hidden_t_value = value_net(
                        torch.from_numpy(s_t).float(),
                        hidden_tm1_value,
                        torch.from_numpy(a_tm1).long(),
                        torch.from_numpy(r_tm1).float(),
                        torch.from_numpy(done_tm1).float(),
                    )  # [B, 1]

                    dist = Categorical(logits=pi_dist_t)
                    a_t = dist.sample()  # [B]
                    log_prob_a_t = dist.log_prob(a_t)  # [B]
                    s_tp1, r_t, done_t, _ = envs.step(a_t.detach().numpy())

                    # Store all t values
                    s_lst.append(s_t)
                    a_lst.append(a_t.detach().numpy())
                    r_lst.append(r_t)
                    done_lst.append(done_t)
                    v_lst.append(v_t.squeeze(-1).detach().numpy())
                    log_prob_a_lst.append(log_prob_a_t.detach().numpy())

                    # Update values
                    s_t = s_tp1
                    a_tm1 = a_lst[-1]
                    r_tm1 = r_lst[-1]
                    done_tm1 = done_lst[-1]
                    hidden_tm1_policy = hidden_t_policy
                    hidden_tm1_value = hidden_t_value

                td_lambda_target, adv_lst = compute_td_lambda_target(
                    r_lst, v_lst, config
                )  # [T, B] for both

                adv_lst = adv_lst.swapaxes(0, 1)  # [B, T]
                td_lambda_target = td_lambda_target.swapaxes(0, 1)  # [B, T]
                s_lst = np.stack(s_lst).swapaxes(0, 1)  # [B, T]
                a_lst = np.stack(a_lst).swapaxes(0, 1)  # [B, T]
                r_lst = np.stack(r_lst).swapaxes(0, 1)  # [B, T]
                done_lst = np.stack(done_lst).swapaxes(0, 1)  # [B, T]
                v_lst = np.stack(v_lst).swapaxes(0, 1)  # [B, T]
                log_prob_a_lst = np.stack(log_prob_a_lst).swapaxes(0, 1)  # [B, T]

                # How to "split" the data so that I can put the meta_episode data for each process into its own MetaEpisodeData object?
                for i in range(config.n_train_processes):
                    meta_episodes[i].append(
                        MetaEpisodeData(
                            s_lst=s_lst[i],
                            a_lst=a_lst[i],
                            r_lst=r_lst[i],
                            done_lst=done_lst[i],
                            v_lst=v_lst[i],
                            adv_lst=adv_lst[i],
                            td_lambda_target=td_lambda_target[i],
                            log_prob_a=log_prob_a_lst[i],
                        )
                    )

        # Log mean rewards per meta episode
        mean_reward = np.mean(
            [round(ep.r_lst.sum(), 2) for proc in meta_episodes for ep in proc]
        )
        print(f"Step #{policy_update_idx}, mean reward mean: {mean_reward}")
        writer.add_scalar("charts/mean_reward", mean_reward, policy_update_idx)

        for opt_epoch in range(config.opt_epochs):
            idxs = np.random.permutation(config.meta_episodes_per_policy_update)
            for i in range(
                0,
                config.meta_episodes_per_policy_update,
                config.meta_episodes_batch_size,
            ):
                idx_batch = idxs[i : i + config.meta_episodes_batch_size]
                policy_optimizer.zero_grad()
                value_optimizer.zero_grad()

                for proc in range(config.n_train_processes):
                    proc_meta_episodes = meta_episodes[proc]
                    proc_meta_episodes_batch = [
                        proc_meta_episodes[idx] for idx in idx_batch
                    ]

                    def get_tensor(field, dtype=None):
                        mb_field = np.stack(
                            list(
                                map(
                                    lambda metaep: getattr(metaep, field),
                                    proc_meta_episodes_batch,
                                )
                            ),
                            axis=0,
                        )
                        if dtype == "long":
                            return torch.from_numpy(mb_field).long()
                        return torch.from_numpy(mb_field).float()

                    mb_s = get_tensor("s_lst", dtype="float")
                    mb_a = get_tensor("a_lst", dtype="long")
                    mb_r = get_tensor("r_lst", dtype="float")
                    mb_done = get_tensor("done_lst", dtype="float")
                    mb_td_lambda_target = get_tensor("td_lambda_target", dtype="float")
                    mb_adv = get_tensor("adv_lst", dtype="float")
                    mb_log_prob_a = get_tensor("log_prob_a", dtype="float")
                    a_dummy = torch.zeros(
                        (config.meta_episodes_batch_size, 1)
                    ).long()  # [B, 1]
                    r_dummy = torch.zeros(
                        (config.meta_episodes_batch_size, 1)
                    )  # [B, 1]
                    done_dummy = torch.ones(
                        (config.meta_episodes_batch_size, 1)
                    )  # [B, 1]

                    prev_action = torch.cat((a_dummy, mb_a[:, :-1]), dim=1)  # [B, T]
                    prev_reward = torch.cat((r_dummy, mb_r[:, :-1]), dim=1)  # [B, T]
                    prev_done = torch.cat(
                        (done_dummy, mb_done[:, :-1]), dim=1
                    )  # [B, T]

                    hidden_policy = policy_net.initial_state(
                        config.meta_episodes_batch_size
                    )  # [B, 512]
                    hidden_value = value_net.initial_state(
                        config.meta_episodes_batch_size
                    )  # [B, 512]

                    # Get policy probabilities
                    pi_dist, _ = policy_net(
                        mb_s,
                        hidden_policy,
                        prev_action,
                        prev_reward,
                        prev_done,
                    )

                    # Get value estimates
                    v, _ = value_net(
                        mb_s,
                        hidden_value,
                        prev_action,
                        prev_reward,
                        prev_done,
                    )

                    # Calculate losses
                    dist = Categorical(logits=pi_dist)
                    entropy = dist.entropy()
                    log_prob_a = dist.log_prob(mb_a)

                    policy_ratios = torch.exp(log_prob_a - mb_log_prob_a)
                    clipped_policy_ratios = torch.clip(
                        policy_ratios, 1 - config.clip_ratio, 1 + config.clip_ratio
                    )
                    surr_1 = mb_adv * policy_ratios
                    surr_2 = mb_adv * clipped_policy_ratios
                    policy_surrogate_objective = torch.mean(torch.min(surr_1, surr_2))

                    policy_loss = -(
                        policy_surrogate_objective
                        + config.entropy_coef * torch.mean(entropy)
                    )
                    value_loss = F.smooth_l1_loss(v.squeeze(-1), mb_td_lambda_target)

                    policy_loss.backward(retain_graph=True)
                    value_loss.backward()

                # After accumulating gradients from all processes, step once
                # This automatically divides by n_train_processes
                for param in policy_net.parameters():
                    if param.grad is not None:
                        param.grad = param.grad / config.n_train_processes

                for param in value_net.parameters():
                    if param.grad is not None:
                        param.grad = param.grad / config.n_train_processes

                policy_optimizer.step()
                value_optimizer.step()

        # Run evaluation
        avg_score, avg_final_episode_score = test(policy_update_idx, policy_net, config)
        writer.add_scalar("charts/test_mean_reward", avg_score, policy_update_idx)
        writer.add_scalar(
            "charts/test_mean_episode_return",
            avg_final_episode_score,
            policy_update_idx,
        )

        # Save checkpoint if enabled
        if config.checkpoint and policy_update_idx % config.checkpoint_frequency == 0:
            checkpoint_dir = f"checkpoints/{run_name}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_path = f"{checkpoint_dir}/checkpoint_{policy_update_idx}.pt"
            torch.save(
                {
                    "policy_update_idx": policy_update_idx,
                    "policy_net_state_dict": policy_net.state_dict(),
                    "value_net_state_dict": value_net.state_dict(),
                    "policy_optimizer_state_dict": policy_optimizer.state_dict(),
                    "value_optimizer_state_dict": value_optimizer.state_dict(),
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

    run_name = f"meta_a2c_braindead__{config.seed}__{config.train_percentage}__{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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

    # Seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Shuffle our level0 puzzles based on config
    shuffle_puzzles(
        braindead.__path__[0],
        config.train_percentage,
        config.test_percentage,
        config.archive_percentage,
        config.seed,
    )

    try:
        envs = ParallelEnv(config)
        main(config, writer, envs)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Cleaning up...")
    finally:
        writer.close()
        if envs:
            envs.close()
        if config.track:
            wandb.finish()
