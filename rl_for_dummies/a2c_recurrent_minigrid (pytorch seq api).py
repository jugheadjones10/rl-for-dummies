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
        self.lstm = nn.LSTM(256, 128)  # input size, hidden size
        self.fc_pi = nn.Linear(128, 7)  # 7 actions for MiniGrid
        self.fc_v = nn.Linear(128, 1)

    def forward(self, x, hidden):
        """Common forward pass for both policy and value networks"""
        # Normalize input
        x = x / 255.0

        # Handle different input formats
        # if len(x.shape) == 3:  # Single observation [H,W,C]
        #     x = x.unsqueeze(0)  # Add batch dimension [1,H,W,C]

        # First we combine seq_len and batch_size into a single dimension so that they become the batch size
        x = x.view(self.seq_len * self.batch_size, *x.shape[2:])

        # Convert from [B,H,W,C] to [B,C,H,W] for Conv2d
        x = x.permute(0, 3, 1, 2)

        # CNN feature extraction
        x = F.relu(self.conv1(x))

        # Flatten for fully connected layers
        x = x.reshape(x.size(0), -1)

        # Shared fully connected layer
        x = F.relu(self.fc1(x))

        # Reshape back to [seq_len, batch_size, features]
        x = x.view(self.seq_len, self.batch_size, -1)

        # Add D * num_layers dimension to hidden state (as per pytorch docs)
        hidden = hidden.unsqueeze(0)  # Shape becomes [1, batch_size, 256]

        # Split hidden state into cell and hidden state
        hidden_cell = hidden.chunk(2, dim=2)

        # x: [seq_len, batch_size, 128]
        x, lstm_hidden = self.lstm(x, hidden_cell)
        # lstm_hidden is a tuple of two tensors: (h_n, c_n)
        # h_n.shape: [1, batch_size, 128]
        # c_n.shape: [1, batch_size, 128]

        # Re-combine cell and hidden state
        lstm_hidden = torch.cat(lstm_hidden, dim=2)
        lstm_hidden = lstm_hidden.squeeze(0)  # Shape becomes [batch_size, 128]

        return x, lstm_hidden

    # x: [seq_len, batch_size, 7, 7, 3]
    # hidden: [batch_size, 256]
    def pi(self, x, hidden, softmax_dim=1):
        """Policy network: returns action probabilities"""
        self.seq_len = x.shape[0]
        self.batch_size = x.shape[1]
        x, lstm_hidden = self.forward(x, hidden)  # x: [seq_len, batch_size, 128]
        # Again combine seq_len and batch_size into a single dimension so that they become the batch size
        x = x.view(self.seq_len * self.batch_size, -1)  # [seq_len * batch_size, 128]
        x = self.fc_pi(x)  # [seq_len * batch_size, 7]
        prob = F.softmax(x, dim=softmax_dim)  # [seq_len * batch_size, 7]
        # Reshape back to [seq_len, batch_size, 7]
        prob = prob.view(self.seq_len, self.batch_size, -1)  # [seq_len, batch_size, 7]
        return prob, lstm_hidden

    def v(self, x, hidden):
        """Value network: returns state value estimate"""
        self.seq_len = x.shape[0]
        self.batch_size = x.shape[1]
        x, lstm_hidden = self.forward(x, hidden)
        # Again combine seq_len and batch_size into a single dimension so that they become the batch size
        x = x.view(self.seq_len * self.batch_size, -1)  # [seq_len * batch_size, 128]
        v = self.fc_v(x)  # [seq_len * batch_size, 1]
        # Reshape back to [seq_len, batch_size, 1]
        v = v.view(self.seq_len, self.batch_size, -1)  # [seq_len, batch_size, 1]
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
    env = gym.make("MiniGrid-DoorKey-5x5-v0", max_episode_steps=500)
    env = ImgObsWrapper(env)
    score = 0.0
    done = False
    num_test = 10

    for _ in range(num_test):
        s, _ = env.reset()
        h_in = torch.zeros(1, 2 * 128, dtype=torch.float)  # (1, 256)
        while not done:
            prob, h_in = model.pi(torch.from_numpy(s).float(), h_in)
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
    # First half is cell state, second half is hidden state
    initial_state = torch.zeros(
        n_train_processes, 2 * 128, dtype=torch.float
    )  # (n_train_processes, 256)
    h_in = initial_state
    while step_idx < max_train_steps:
        s_lst, a_lst, r_lst, v_lst, mask_lst = list(), list(), list(), list(), list()
        h_initial = h_in
        for _ in range(update_interval):
            s = torch.from_numpy(s).float()
            prob, h_out = model.pi(
                s.unsqueeze(0), h_in
            )  # We add seq len 1 dimension to state
            prob = prob.squeeze(0)  # [seq_len, batch_size, 7] -> [batch_size, 7]
            v = model.v(s.unsqueeze(0), h_in).detach().numpy()  # (num_envs, 1)
            v = v.squeeze()  # [seq_len, batch_size, 1] -> [batch_size]
            a = Categorical(prob).sample().numpy()  # (num_envs,)
            s_prime, r, done, info = envs.step(a)

            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            v_lst.append(v)
            mask_lst.append(1 - done)

            mask = torch.tensor(mask_lst[-1]).unsqueeze(-1)
            h_in = h_out * mask + (1 - mask) * initial_state
            s = s_prime
            step_idx += 1

        s_final = torch.from_numpy(s_prime).float()
        v_final = model.v(s_final.unsqueeze(0), h_in).detach().clone().numpy()
        v_final = v_final.squeeze(0)  # [seq_len, batch_size, 1] -> [batch_size, 1]
        td_target = compute_target(
            v_final, r_lst, mask_lst
        )  # (update_interval, num_envs)

        # v_lst: (update_interval, num_envs, 1)
        v_vec = torch.tensor(v_lst)  # (update_interval, num_envs)
        s_vec = torch.stack(s_lst)  # (update_interval, num_envs, 7, 7, 3)
        a_vec = torch.tensor(a_lst)  # (update_interval, num_envs)
        mask_lst = torch.tensor(mask_lst)  # (update_interval, num_envs)

        advantage = td_target - v_vec  # (update_interval, num_envs)

        policy_loss = []
        value_loss = []
        entropy_loss = []  # New list to store entropy values
        entropy_coef = 0.01  # Coefficient for entropy term (adjust as needed)
        h_in_ = h_initial.detach()

        # Now use the fact that I now support seq len for my Actor Critic model to replace the loop
        prob, _ = model.pi(s_vec, h_in_)  # (num_envs, 7)
        pi_a = torch.gather(prob, 2, a_vec.unsqueeze(-1)).squeeze(-1)
        policy_loss = -(torch.log(pi_a) * advantage).mean()
        value_loss = F.smooth_l1_loss(model.v(s_vec, h_in_).squeeze(-1), td_target)
        # entropy_loss = -(prob * torch.log(prob + 1e-10)).sum(dim=1).mean()

        # Skip entropy for now
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_idx % PRINT_INTERVAL == 0:
            test(step_idx, model)

    envs.close()
