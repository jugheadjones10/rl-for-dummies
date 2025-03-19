import random

import gymnasium as gym
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
n_train_processes = 3
learning_rate = 0.0002
update_interval = 20
gamma = 0.98
max_train_steps = 10**7
entropy_coef = 0.01
PRINT_INTERVAL = update_interval * 100

SEED = 42


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=1)

        # Calculate output size after convolution: 5-2+1 = 4
        conv_output_size = 4 * 4 * 16

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.fc_pi = nn.Linear(64, 7)  # 7 actions for MiniGrid
        self.fc_v = nn.Linear(64, 1)

    def forward(self, x):
        """Common forward pass for both policy and value networks"""
        # Normalize input
        x = x / 255.0

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
    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    env.action_space.seed(SEED)

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
    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    score = 0.0
    done = False
    num_test = 10

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
    print(f"Step # :{step_idx}, avg score : {score/num_test:.1f}")

    env.close()


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    envs = ParallelEnv(n_train_processes)

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    step_idx = 0
    s = envs.reset()
    while step_idx < max_train_steps:
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
            test(step_idx, model)

    envs.close()
