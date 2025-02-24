import gymnasium as gym
import numpy as np

from probe_envs.shortcorridor import ShortCorridorEnv


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


# Will export the "make_env function?"
def shortcorridor_make_env(env_id, render_mode, **env_kwargs):
    env = ShortCorridorEnv()
    env = OneHotWrapper(env)
    return env
