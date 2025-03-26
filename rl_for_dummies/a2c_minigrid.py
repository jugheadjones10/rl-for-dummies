import random

import gymnasium as gym
import matplotlib.pyplot as plt
import minigrid  # noqa
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper  # noqa
from torch.distributions import Categorical

# Hyperparameters
n_train_processes = 4
learning_rate = 0.0002
update_interval = 10
gamma = 0.98
max_train_steps = 10**7
entropy_coef = 0.1

MINIGRID_ENV = "MiniGrid-Empty-5x5-v0"
PRINT_INTERVAL = update_interval * 10
SEED = 42


# Add a wrapper that restricts the action space to only the used actions (Actions 0, 1, 2)
class RestrictedActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3)


class SimplifiedStateWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        """Reset the environment and return custom observation"""
        obs, info = self.env.reset(**kwargs)
        return self._get_observation(obs), info

    def step(self, action):
        """Take a step and return custom observation"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._get_observation(obs), reward, terminated, truncated, info

    def _get_observation(self, obs):
        """Convert Minigrid observation to custom format"""
        # Initialize empty 5x5 grid
        grid = np.zeros((2, 5, 5), dtype=np.uint8)

        obs = obs.transpose(2, 0, 1)
        object_idx = obs[0]

        wall_pos = np.where(object_idx == 2)
        grid[0][wall_pos[0], wall_pos[1]] = 3

        agent_pos = np.where(object_idx == 10)
        grid[0][agent_pos[0], agent_pos[1]] = 1

        goal_pos = np.where(object_idx == 8)
        grid[0][goal_pos[0], goal_pos[1]] = 2

        # Second grid channel is for direction
        grid[1][agent_pos[0], agent_pos[1]] = self.unwrapped.agent_dir + 1

        # Return dictionary with grid and direction
        return grid.transpose(1, 2, 0)


# Gym wrapper that removes the color channel and state channel
class SimplifiedObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = gym.spaces.Box(
        #     low=0, high=10, shape=(5, 5, 1), dtype=np.uint8
        # )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._get_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._get_observation(obs), reward, terminated, truncated, info

    def _get_observation(self, obs):
        return np.expand_dims(obs[:, :, 0], axis=-1)


def get_initial_state():
    env = gym.make(MINIGRID_ENV)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    env = RestrictedActionWrapper(env)
    # env = SimplifiedObsWrapper(env)
    env = SimplifiedStateWrapper(env)
    s, _ = env.reset()
    env.close()
    return s


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(2, 8, kernel_size=2, stride=1)

        # Calculate output size after convolution:
        conv_output_size = 4 * 4 * 8

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.fc_pi = nn.Linear(64, 3)  # 3 actions for MiniGrid
        self.fc_v = nn.Linear(64, 1)

    def forward(self, x):
        """Common forward pass for both policy and value networks"""

        # Handle different input formats
        if len(x.shape) == 3:  # Single observation [H,W,C]
            x = x.unsqueeze(0)  # Add batch dimension [1,H,W,C]

        # Convert from [B,H,W,C] to [B,C,H,W] for Conv2d
        x = x.permute(0, 3, 1, 2)

        # CNN feature extraction
        x = F.relu(self.conv1(x))

        # Flatten for fully connected layers
        x = x.reshape(x.size(0), -1)

        # Shared fully connected layer
        x = F.relu(self.fc1(x))
        return x

    def pi(self, x, softmax_dim=1):
        """Policy network: returns action probabilities"""
        features = self.forward(x)
        action_logits = self.fc_pi(features)
        return F.softmax(action_logits, dim=softmax_dim)

    def v(self, x):
        """Value network: returns state value estimate"""
        features = self.forward(x)
        return self.fc_v(features)


def worker(worker_id, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging
    env = gym.make(MINIGRID_ENV)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    env = RestrictedActionWrapper(env)
    env = SimplifiedStateWrapper(env)
    # env = SimplifiedObsWrapper(env)
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


def visualize_action_probs(model, get_initial_state):
    # Create a copy of your initial state
    s_ = get_initial_state().copy()
    s_[1, 1, 0] = 1  # Put agent at (1,1) initially (just as your snippet does)

    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    fig.suptitle("Action Probability Distribution for Each Cell")

    # Loop through the 5×5 grid
    for i in range(5):
        for j in range(5):
            # Place the agent temporarily at (i,j)
            s_[i, j, 0] = 10

            # Compute action probabilities
            with torch.no_grad():  # Turn off grad to speed up
                prob_ = model.pi(torch.from_numpy(s_).float())
            prob_values = prob_.numpy().flatten()

            # Revert (i, j) position so it doesn't affect next cell
            s_[i, j, 0] = 1

            ax = axes[i, j]

            actions = ["Left", "Right", "Forward"]
            base_fontsize = 8
            scale = 10

            for a_idx, action_name in enumerate(actions):
                p = prob_values[a_idx]
                fontsize = base_fontsize + p**4 * scale
                y_offset = 0.75 - 0.25 * a_idx
                ax.text(
                    0.5,
                    y_offset,
                    f"{action_name}: {p:.2f}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    transform=ax.transAxes,
                )

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def test(step_idx, model):
    env = gym.make(MINIGRID_ENV)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    env = RestrictedActionWrapper(env)
    env = SimplifiedStateWrapper(env)
    # env = SimplifiedObsWrapper(env)
    env.action_space.seed(SEED)
    score = 0.0
    done = False
    num_test = 5
    for _ in range(num_test):
        s, _ = env.reset()
        while not done:
            prob = model.pi(torch.from_numpy(s).float())
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

    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        G = r + gamma * G * mask
        td_target.append(G)

    return torch.tensor(td_target[::-1]).float()


if __name__ == "__main__":
    # Seeding
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    envs = ParallelEnv(n_train_processes)
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    step_idx = 0
    s = envs.reset()
    ep_returns = []

    while step_idx < 30000:
        s_lst, a_lst, r_lst, mask_lst = list(), list(), list(), list()
        for _ in range(update_interval):
            prob = model.pi(torch.from_numpy(s).float())
            a = Categorical(prob).sample().numpy()
            s_prime, r, done, info = envs.step(a)

            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            mask_lst.append(1 - done)

            s = s_prime
            step_idx += 1

        s_final = torch.from_numpy(s_prime).float()
        v_final = model.v(s_final).detach().clone().numpy()
        td_target = compute_target(v_final, r_lst, mask_lst)

        td_target_vec = td_target.reshape(-1)
        s_vec = torch.tensor(s_lst).float().reshape(-1, *s_lst[0].shape[1:])
        a_vec = torch.tensor(a_lst).reshape(-1).unsqueeze(1)
        advantage = td_target_vec - model.v(s_vec).reshape(-1)

        pi = model.pi(s_vec)
        pi_a = pi.gather(1, a_vec).reshape(-1)

        # Add entropy loss
        dist = Categorical(pi)
        entropy = dist.entropy().mean()

        loss = (
            -(torch.log(pi_a) * advantage.detach()).mean()
            + F.smooth_l1_loss(model.v(s_vec).reshape(-1), td_target_vec)
            - entropy_coef * entropy
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_idx % PRINT_INTERVAL == 0:
            avg_score = test(step_idx, model)
            ep_returns.append(avg_score)

        if step_idx % 10000 == 0:
            # Visualise the policy in each state
            # For each grid position, plot the action probabilities
            # visualize_action_probs(model, get_initial_state)
            pass

            # s_ = get_initial_state()
            # s_[1, 1, 0] = 1
            # for i in range(5):
            #     for j in range(5):
            #         s_[i, j, 0] = 10
            #         prob_ = model.pi(torch.from_numpy(s_).float())
            #         print(f"State ({i}, {j}): {prob_}")
            #         s_[i, j, 0] = 1

    plt.plot(ep_returns)
    plt.show()
