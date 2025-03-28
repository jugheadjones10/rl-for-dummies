import os
import random
from dataclasses import dataclass

import minigrid  # noqa
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper  # noqa
from pushworld.config import BENCHMARK_PUZZLES_PATH
from torch.distributions import Categorical

from envs.pushworld import PushWorldEnv

# Hyperparameters
n_train_processes = 2
learning_rate = 0.0002
update_interval = 10
gamma = 0.98
max_train_steps = 10**7
entropy_coef = 0.1

PRINT_INTERVAL = update_interval * 10
SEED = 42


def make_env(render_mode=None):
    env = PushWorldEnv(
        max_steps=50,
        puzzle_path=os.path.join(
            BENCHMARK_PUZZLES_PATH,
            "level0",
            "mini",
        ),
        render_mode=render_mode,
    )
    return env


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(4, 8, kernel_size=2, stride=1)

        # Calculate output size after convolution:
        conv_output_size = 6 * 6 * 8

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.lstm_cell = nn.LSTMCell(294, 64)
        self.fc_pi = nn.Linear(64, 4)  # 4 actions for PushWorld
        self.fc_v = nn.Linear(64, 1)

    def initial_state(self):
        return torch.zeros(
            n_train_processes, 2 * 64, dtype=torch.float
        )  # (n_train_processes, 128)

    def forward(self, x, hidden, prev_action, prev_reward, prev_done):
        """Common forward pass for both policy and value networks"""
        # CNNs can only have one batch dimension
        # So if our states is of the form [B, T, H, W, C], we need to combine the B and T dimensions
        # first, and then re-separate them later.

        # Possible x shapes:
        # [H, W, C] (No batch dimension: this is during single env testing)
        # [B, T, H, W, C] (Batch and time dimensions: happens during training)

        # Possible prev_action/prev_reward/prev_done shapes:
        # [B] (No time dimension: this is during data collection)
        # [B, T] (Time dimension: happens during training)

        reshaped = False
        if len(x.shape) == 5:
            # For x: [B, T, H, W, C] -> [B*T, H, W, C]
            # For prevs: [B] -> [B*T]
            batch_size = x.shape[0]
            time_steps = x.shape[1]
            x = x.reshape(batch_size * time_steps, *x.shape[2:])
            prev_action = prev_action.reshape(batch_size * time_steps)
            prev_reward = prev_reward.reshape(batch_size * time_steps)
            prev_done = prev_done.reshape(batch_size * time_steps)
            reshaped = True
        elif len(x.shape) == 3:
            # [H, W, C] -> [1, H, W, C] (Because CNN requires batch dimension)
            x = x.unsqueeze(0)

        # Convert from [B, H, W, C] to [B, C, H, W] for Conv2d
        x = x.permute(0, 3, 1, 2)

        # CNN feature extraction
        x = F.relu(self.conv1(x))

        # Flatten for fully connected layers
        x = x.reshape(x.size(0), -1)  # [B*T, 6*6*8]

        # To the final dimension, I want to concat one-hot encoded prev_action, prev_reward, and prev_done
        prev_action_one_hot = F.one_hot(prev_action, num_classes=4)
        x = torch.cat(
            (
                x,
                prev_action_one_hot,
                prev_reward.unsqueeze(-1),
                prev_done.unsqueeze(-1),
            ),
            dim=-1,
        )

        # Re-separate the batch and time dimensions
        if x.shape[0] == 1:
            # [1, 64] -> [1, 1, 64]
            x = x.unsqueeze(0)

        if reshaped:
            # For x: [B*T, 64] -> [B, T, 64]
            # For prevs: [B*T] -> [B, T]
            x = x.reshape(batch_size, time_steps, -1)
            prev_action = prev_action.reshape(batch_size, time_steps)
            prev_reward = prev_reward.reshape(batch_size, time_steps)
            prev_done = prev_done.reshape(batch_size, time_steps)
        else:
            # [B, 64] -> [B, 1, 64]
            x = x.unsqueeze(1)

        if len(hidden.shape) == 1:
            # [64] -> [1, 64]
            hidden = hidden.unsqueeze(0)  # Add batch dimension [1, 64]

        # Now we step through time steps, using LSTM Cell
        T = x.shape[1]
        features_by_timestep = []
        for t in range(0, T):
            h_n, c_n = hidden.chunk(2, dim=1)
            x_t = x[:, t, :]
            h_n, c_n = self.lstm_cell(x_t, (h_n, c_n))
            features_by_timestep.append(h_n)
            hidden = torch.cat((h_n, c_n), dim=1)

        features = torch.stack(features_by_timestep, dim=1)  # [B, T, 64]
        if T == 1:
            features = features.squeeze(1)  # [B, 64]

        # If T == 1, features is [B, 64]
        # Otherwise, features is [B, T, 64]
        # hidden is just [B, 128]
        return features, hidden

    # x: [batch_size, 7, 7, 3]
    # hidden: [batch_size, 64]
    def pi(self, x, hidden, prev_action, prev_reward, prev_done, softmax_dim=1):
        """Policy network: returns action probabilities"""
        features, memory = self.forward(x, hidden, prev_action, prev_reward, prev_done)
        action_logits = self.fc_pi(features)  # [batch_size, 4] or [batch_size, T, 4]
        prob = F.softmax(action_logits, dim=-1)
        return prob, memory

    def v(self, x, hidden, prev_action, prev_reward, prev_done):
        """Value network: returns state value estimate"""
        features, _ = self.forward(x, hidden, prev_action, prev_reward, prev_done)
        v = self.fc_v(features)
        return v


def worker(worker_id, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging
    env = make_env()
    env.action_space.seed(SEED + worker_id)

    while True:
        cmd, data = worker_end.recv()
        if cmd == "step":
            ob, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            if done:
                ob, _ = env.reset()
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
    def __init__(self, n_train_processes):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(
            zip(master_ends, worker_ends)
        ):
            p = mp.Process(target=worker, args=(worker_id, master_end, worker_end))
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


def test(step_idx, model):
    env = make_env()
    env.action_space.seed(SEED)
    score = 0.0
    done = False
    num_test = 5
    for _ in range(num_test):
        s, _ = env.reset()
        initial_state = torch.zeros(2 * 64, dtype=torch.float)
        h_in = initial_state
        while not done:
            prob, h_in = model.pi(torch.from_numpy(s).float(), h_in)
            a = Categorical(prob).sample().item()
            s_prime, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            s = s_prime
            score += r
        done = False
    avg_score = round(score / num_test, 2)
    print(f"Step # :{step_idx}, avg score : {avg_score}")

    env.close()
    return avg_score


def compute_target(v_final, r_lst, mask_lst):
    G = v_final.reshape(-1)
    td_target = list()

    # Note that in the meta-episodic setting of RL^2, the objective is
    # to maximize the expected discounted return of the meta-episode,
    # so we do not utilize the usual 'done' masking in this function.
    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        # G = r + gamma * G * mask
        G = r + gamma * G
        td_target.append(G)

    return torch.tensor(td_target[::-1]).float()


num_policy_updates = 100
meta_episodes_per_policy_update = 10
meta_episode_length = 100


# Data class object to store a meta episode data
@dataclass
class MetaEpisodeData:
    s_lst: np.ndarray
    a_lst: np.ndarray
    r_lst: np.ndarray
    done_lst: np.ndarray
    log_prob_lst: np.ndarray
    v_lst: np.ndarray
    adv_lst: np.ndarray


if __name__ == "__main__":
    # Seeding
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    envs = ParallelEnv(n_train_processes)
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for policy_update_idx in range(num_policy_updates):
        meta_episodes = []
        for _ in range(meta_episodes_per_policy_update):
            s_lst, a_lst, r_lst, done_lst, log_prob_lst, v_lst = (
                list(),
                list(),
                list(),
                list(),
                list(),
                list(),
            )

            s_t = envs.reset()  # [B, 7, 7, 3]
            a_tm1 = np.zeros((n_train_processes,))  # [B]
            r_tm1 = np.zeros((n_train_processes,))  # [B]
            done_tm1 = np.zeros((n_train_processes,))  # [B]

            # Initial hidden state
            hidden_tm1 = model.initial_state()  # [B, 128]

            for t in range(0, meta_episode_length):
                prob_t, hidden_t = model.pi(
                    torch.from_numpy(s_t).float(),
                    hidden_tm1,
                    torch.from_numpy(a_tm1).long(),
                    torch.from_numpy(r_tm1).float(),
                    torch.from_numpy(done_tm1).float(),
                )  # [B, 4]

                v_t = model.v(
                    torch.from_numpy(s_t).float(),
                    hidden_tm1,
                    torch.from_numpy(a_tm1).long(),
                    torch.from_numpy(r_tm1).float(),
                    torch.from_numpy(done_tm1).float(),
                )  # [B, 1]

                a_t = Categorical(prob_t).sample()  # [B]
                log_prob_t = torch.log(prob_t.gather(1, a_t.reshape(-1, 1)))  # [B]

                s_tp1, r_t, done_t, _ = envs.step(a_t.numpy())

                # Store all t values
                s_lst.append(s_t)
                a_lst.append(a_t.detach().numpy())
                r_lst.append(r_t)
                done_lst.append(done_t)
                log_prob_lst.append(log_prob_t.detach().numpy())
                v_lst.append(v_t.squeeze(-1).detach().numpy())

                # Update values
                s_t = s_tp1
                a_tm1 = a_lst[-1]
                r_tm1 = r_lst[-1]
                done_tm1 = done_lst[-1]
                hidden_tm1 = hidden_t

            # We now have our meta-episode batch
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
            )
            td_target = compute_target(
                v_final, r_lst, done_lst
            )  # [meta_episode_length, B]

            v_vec = torch.tensor(v_lst)  # [meta_episode_length, B]
            # Both td_target and v_vec here are tensors. But do we need them to be?
            adv_lst = (td_target - v_vec).detach().numpy()  # [meta_episode_length, B]
            adv_lst = adv_lst.swapaxes(0, 1)  # [B, meta_episode_length]

            s_lst = np.stack(s_lst).swapaxes(0, 1)  # [B, meta_episode_length, 7, 7, 3]
            a_lst = np.stack(a_lst).swapaxes(0, 1)  # [B, meta_episode_length]
            r_lst = np.stack(r_lst).swapaxes(0, 1)  # [B, meta_episode_length]
            done_lst = np.stack(done_lst).swapaxes(0, 1)  # [B, meta_episode_length]
            log_prob_lst = np.stack(log_prob_lst).swapaxes(
                0, 1
            )  # [B, meta_episode_length]
            v_lst = np.stack(v_lst).swapaxes(0, 1)  # [B, meta_episode_length]

            meta_episode_data = MetaEpisodeData(
                s_lst=s_lst,
                a_lst=a_lst,
                r_lst=r_lst,
                done_lst=done_lst,
                log_prob_lst=log_prob_lst,
                v_lst=v_lst,
                adv_lst=adv_lst,
            )
            meta_episodes.append(meta_episode_data)

        # Now we do policy update using meta_episode_data
        # We randomly loop through the meta_episodes.
        # Remember, each meta_episode contains a batch of size num_processes.
        idxs = np.random.permutation(meta_episodes_per_policy_update)
        for i in range(0, meta_episodes_per_policy_update):
            meta_episode = meta_episodes[idxs[i]]

            # Compute losses
            mb_s = torch.from_numpy(meta_episode.s_lst).float()  # [B, T, 7, 7, 3]
            mb_a = torch.from_numpy(meta_episode.a_lst).long()  # [B, T]
            mb_r = torch.from_numpy(meta_episode.r_lst).float()  # [B, T]
            mb_done = torch.from_numpy(meta_episode.done_lst).float()  # [B, T]
            mb_log_prob = torch.from_numpy(meta_episode.log_prob_lst).float()  # [B, T]
            mb_v = torch.from_numpy(meta_episode.v_lst).float()  # [B, T]
            mb_adv = torch.from_numpy(meta_episode.adv_lst).float()  # [B, T]

            a_dummy = torch.zeros((n_train_processes, 1)).long()
            r_dummy = torch.zeros((n_train_processes, 1))
            done_dummy = torch.zeros((n_train_processes, 1))

            prev_action = torch.cat((a_dummy, mb_a[:, :-1]), dim=1)
            prev_reward = torch.cat((r_dummy, mb_r[:, :-1]), dim=1)
            prev_done = torch.cat((done_dummy, mb_done[:, :-1]), dim=1)

            hidden = model.initial_state()

            prob, _ = model.pi(
                mb_s,
                hidden,
                prev_action,
                prev_reward,
                prev_done,
            )

            v = model.v(
                mb_s,
                hidden,
                prev_action,
                prev_reward,
                prev_done,
            )

            entropy = Categorical(prob).entropy()

            # Use prob to calculate log prob again using mb_a
            log_prob_a = torch.log(prob.gather(1, mb_a.unsqueeze(-1)).squeeze(-1))

            policy_loss = -torch.mean(log_prob_a * mb_adv)
            value_loss = F.smooth_l1_loss(v.squeeze(-1), mb_v)
            entropy_loss = -torch.mean(entropy)

            loss = policy_loss - entropy_coef * entropy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    envs.close()
    # plt.plot(ep_returns)
    # plt.show()
