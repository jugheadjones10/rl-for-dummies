import gymnasium as gym
from gymnasium import spaces


class OneStepRewardEnv(gym.Env):
    """
    A custom Gymnasium environment for testing with the following properties:

      - Two discrete actions: 0 and 1.
      - One timestep per episode.
      - Action 0 gives a reward of +1, and action 1 gives a reward of -1.

    This simple setup is ideal for testing whether an agent learns to prefer action 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # There are only 2 actions: 0 and 1.
        self.action_space = spaces.Discrete(2)
        # The observation space is trivial in this environment.
        self.observation_space = spaces.Discrete(1)

    def reset(self, seed=None, options=None):
        """
        Resets the environment state and returns the initial observation along with an info dictionary.
        Since the environment has only one step per episode, we always return 0 as the observation.
        """
        super().reset(seed=seed)
        observation = 0
        info = {}
        return observation, info

    def step(self, action):
        """
        Executes one timestep within the environment given an action.

        Action:
            - 0: Returns reward +1.
            - 1: Returns reward -1.

        Returns:
            observation (int): the observation after stepping (always 0 in this setup)
            reward (int): +1 for action 0, -1 for action 1.
            terminated (bool): True indicating the episode's natural termination.
            truncated (bool): False because this environment doesn't apply step truncation.
            info (dict): Additional debugging information (empty here).
        """
        # Validate the action
        if action not in [0, 1]:
            raise ValueError(f"Invalid action: {action}. Expected 0 or 1.")

        reward = 1 if action == 0 else -1

        # In Gymnasium, we differentiate between a natural episode termination (terminated)
        # and a forced truncation (truncated).
        terminated = True  # The episode ends naturally after one step.
        truncated = False  # There's no truncation here.

        observation = 0
        info = {}
        return observation, reward, terminated, truncated, info
