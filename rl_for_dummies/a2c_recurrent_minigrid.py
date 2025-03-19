import random

import gymnasium as gym
import minigrid  # noqa
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from minigrid.wrappers import ImgObsWrapper  # noqa
from torch.distributions import Categorical

# Hyperparameters
n_train_processes = 3
learning_rate = 0.0002
update_interval = 10
gamma = 0.98
max_train_steps = 60000
PRINT_INTERVAL = update_interval * 100
entropy_coef = 0.01
SEED = 42


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)

        # Calculate output size after convolution: 7-3+1 = 5
        conv_output_size = 5 * 5 * 16

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.lstm = nn.LSTM(256, 128)
        self.lstm_cell = nn.LSTMCell(256, 128)
        self.fc_pi = nn.Linear(128, 7)  # 7 actions for MiniGrid
        self.fc_v = nn.Linear(128, 1)

    def forward(self, x, hidden):
        """Common forward pass for both policy and value networks"""
        # Normalize input
        x = x / 255.0

        # Handle different input formats
        if len(x.shape) == 3:  # Single observation [H,W,C]
            x = x.unsqueeze(0)  # Add batch dimension [1,H,W,C]
        if len(hidden.shape) == 1:
            hidden = hidden.unsqueeze(0)  # Add batch dimension [1,256]

        # Convert from [B,H,W,C] to [B,C,H,W] for Conv2d
        x = x.permute(0, 3, 1, 2)

        # CNN feature extraction
        x = F.relu(self.conv1(x))

        # Flatten for fully connected layers
        x = x.reshape(x.size(0), -1)

        # Shared fully connected layer
        x = F.relu(self.fc1(x))

        # Split hidden state into cell and hidden state
        # We call the hidden state and cell state tuple memory
        memory = hidden.chunk(2, dim=1)

        # hidden_state is (h_n, c_n)
        memory = self.lstm_cell(x, memory)
        embedding = memory[0]

        # Re-combine cell and hidden state
        memory = torch.cat(memory, dim=1)

        return embedding, memory

    # x: [batch_size, 7, 7, 3]
    # hidden: [batch_size, 256]
    def pi(self, x, hidden, softmax_dim=1):
        """Policy network: returns action probabilities"""
        embedding, memory = self.forward(x, hidden)
        x = self.fc_pi(embedding)  # [batch_size, 7]
        prob = F.softmax(x, dim=softmax_dim)
        return prob, memory

    def v(self, x, hidden):
        """Value network: returns state value estimate"""
        embedding, _ = self.forward(x, hidden)
        v = self.fc_v(embedding)
        return v


def worker(worker_id, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging
    env = gym.make("MiniGrid-DoorKey-5x5-v0")
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
    env = ImgObsWrapper(env)
    score = 0.0
    done = False
    num_test = 10

    for _ in range(num_test):
        s, _ = env.reset()
        initial_state = torch.zeros(2 * 128, dtype=torch.float)
        h_in = initial_state
        while not done:
            prob, h_in = model.pi(torch.from_numpy(s).float(), h_in)
            a = Categorical(prob).sample().item()
            s_prime, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            if done:
                h_in = initial_state
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
    torch.autograd.set_detect_anomaly(True)

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
    # First half is hidden state, second half is cell state
    initial_state = torch.zeros(
        n_train_processes, 2 * 128, dtype=torch.float
    )  # (n_train_processes, 256)
    h_in = initial_state
    while step_idx < max_train_steps:
        s_lst, a_lst, r_lst, v_lst, mask_lst = list(), list(), list(), list(), list()
        # If h_in is the last hidden output of the previous iterations of the update_interval
        h_initial = h_in.detach()
        with torch.no_grad():
            for _ in range(update_interval):
                s = torch.from_numpy(s).float()
                prob, h_out = model.pi(s, h_in)
                v = model.v(s, h_in).numpy()  # (num_envs, 1)
                a = Categorical(prob).sample().numpy()  # (num_envs,)

                s_prime, r, done, info = envs.step(a)

                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                v_lst.append(v)
                mask_lst.append(1 - done)

                mask = torch.tensor(mask_lst[-1], dtype=torch.float32).unsqueeze(-1)
                # Reset hidden state for next iteration if current state is done
                h_in = h_out * mask

                s = s_prime
                step_idx += 1

        s_final = torch.from_numpy(s_prime).float()
        v_final = model.v(s_final, h_in).detach().numpy()
        td_target = compute_target(
            v_final, r_lst, mask_lst
        ).detach()  # (update_interval, num_envs)

        # v_lst: (update_interval, num_envs, 1)
        v_vec = torch.tensor(v_lst).squeeze(-1)  # (update_interval, num_envs)
        s_vec = torch.stack(s_lst)  # (update_interval, num_envs, 7, 7, 3)
        a_vec = torch.tensor(a_lst)  # (update_interval, num_envs)

        advantage = (td_target - v_vec).detach()  # (update_interval, num_envs)

        h_in_ = h_initial
        total_loss = torch.tensor(0.0)
        for i in range(update_interval):
            prob, h_out_ = model.pi(s_vec[i], h_in_)  # (num_envs, 7)
            pi_a = prob.gather(1, a_vec[i].reshape(-1, 1)).reshape(-1)

            dist = Categorical(probs=prob)
            entropy = dist.entropy().mean()
            policy_loss = -(torch.log(pi_a) * advantage[i]).mean()
            value_loss = F.smooth_l1_loss(
                model.v(s_vec[i], h_in_).reshape(-1), td_target[i]
            ).mean()

            loss = policy_loss - entropy * entropy_coef + value_loss
            total_loss += loss

            mask = torch.tensor(mask_lst[i], dtype=torch.float32).unsqueeze(-1)
            h_in_ = h_out_ * mask

        total_loss /= update_interval
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step_idx % PRINT_INTERVAL == 0:
            test(step_idx, model)

    envs.close()
