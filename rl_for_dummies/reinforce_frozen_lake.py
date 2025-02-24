import random
import time  # <-- Import time for benchmarking

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
SEED = 42


# Convert Frozen Lake discrete observation space to a one-hot encoded vector.
class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.n = env.observation_space.n
        self.observation_space = gym.spaces.Box(0, 1, (self.n,), dtype=np.float32)

    def observation(self, obs):
        one_hot = np.zeros(self.n, dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 4)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        loss = torch.tensor(0.0)
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss += torch.log(prob) * R
        loss /= len(self.data)
        loss = -loss
        loss.backward()
        self.optimizer.step()
        self.data = []


def main():
    # Seed all random sources for determinism.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env = gym.make("FrozenLake-v1", is_slippery=False)
    env = OneHotWrapper(env)
    pi = Policy()
    score = 0.0
    time_taken = 0.0
    print_interval = 20

    for n_epi in range(10000):
        start_time = time.time()  # Start timing at the beginning of the episode

        s, _ = env.reset()
        done = False
        while not done:  # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, truncated, info = env.step(a.item())
            pi.put_data((r, prob[a]))
            s = s_prime
            score += r

        pi.train_net()
        time_taken += time.time() - start_time

        if score / print_interval == 0.95:
            print(f"It took {n_epi} episodes to reach 95% of the max score")
            break

        if n_epi % print_interval == 0 and n_epi != 0:
            print(
                f"# of episode :{n_epi}, avg score : {score / print_interval}, avg episode time: {time_taken / print_interval:.4f} sec"
            )
            score = 0.0
            time_taken = 0.0
    env.close()


if __name__ == "__main__":
    main()
