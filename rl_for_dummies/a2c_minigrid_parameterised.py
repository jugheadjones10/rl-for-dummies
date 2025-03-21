import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import minigrid  # noqa
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from minigrid.wrappers import ImgObsWrapper  # noqa
from torch.distributions import Categorical


@dataclass
class TrainingConfig:
    """Configuration for A2C training on MiniGrid environments"""

    # Environment settings
    env_id: str = "MiniGrid-Empty-5x5-v0"

    # Training hyperparameters
    n_train_processes: int = 4
    learning_rate: float = 0.0002
    update_interval: int = 10
    gamma: float = 0.98
    max_train_steps: int = 10**7
    entropy_coef: float = 0.1

    # Output settings
    output_dir: Path = Path("./results")
    experiment_name: Optional[str] = None
    print_interval: int = 100

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        if self.experiment_name is None:
            self.experiment_name = f"vanilla-a2c_{self.env_id}_{self.seed}"

        self.output_path = self.output_dir / self.experiment_name
        os.makedirs(self.output_path, exist_ok=True)

        # If results.csv exists, delete it
        if os.path.exists(self.output_path / "results.csv"):
            os.remove(self.output_path / "results.csv")


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=1)

        # Calculate output size after convolution:
        conv_output_size = 6 * 6 * 16

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.fc_pi = nn.Linear(64, 7)  # 7 actions for MiniGrid
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


def worker(worker_id, master_end, worker_end, config):
    master_end.close()  # Forbid worker to use the master end for messaging
    env = gym.make(config.env_id)
    env = ImgObsWrapper(env)
    env.action_space.seed(config.seed + worker_id)

    while True:
        cmd, data = worker_end.recv()
        if cmd == "step":
            ob, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            if done:
                ob, _ = env.reset()
            worker_end.send((ob, reward, done, info))
        elif cmd == "reset":
            ob, _ = env.reset(seed=config.seed + worker_id)
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
    env = gym.make(config.env_id)
    env = ImgObsWrapper(env)
    env.action_space.seed(config.seed)
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


def compute_target(v_final, r_lst, mask_lst, gamma):
    G = v_final.reshape(-1)
    td_target = list()

    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        G = r + gamma * G * mask
        td_target.append(G)

    return torch.tensor(td_target[::-1]).float()


def train(config):
    # Seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    envs = ParallelEnv(config)
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Start training
    step_idx = 0
    s = envs.reset()
    ep_returns = []

    while step_idx < config.max_train_steps:
        s_lst, a_lst, r_lst, mask_lst = list(), list(), list(), list()
        for _ in range(config.update_interval):
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
        td_target = compute_target(v_final, r_lst, mask_lst, config.gamma)

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
            - config.entropy_coef * entropy
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_idx % config.print_interval == 0:
            avg_score = test(step_idx, model, config)
            ep_returns.append(avg_score)
            write_result(step_idx, avg_score, config)
            # if avg_score >= 0.85:
            #     print(f"Step #{step_idx} finished with avg score {avg_score}")
            #     break

    envs.close()
    return ep_returns


# Create result csv file if it doesn't exit.
# If it exists, use it
# Simply append avg_score as a new row
def write_result(step_idx, avg_score, config):
    result_path = config.output_path / "results.csv"
    if not result_path.exists():
        with open(result_path, "w") as f:
            f.write("step,avg_score\n")
    with open(result_path, "a") as f:
        f.write(f"{step_idx},{avg_score}\n")


if __name__ == "__main__":
    # Parse command line arguments using tyro
    config = tyro.cli(TrainingConfig)

    # Run training
    results = train(config)
