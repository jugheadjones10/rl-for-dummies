import gymnasium as gym
from gymnasium import spaces


class ShortCorridorEnv(gym.Env):
    """
    A simple corridor environment example:

    The corridor layout:

      -----------------
      | S | r | n | G |
      -----------------

    Where:
      S - start state (normal state).
      r - reverse state (actions are reversed; e.g., left becomes right).
      n - normal state (actions work as expected).
      G - goal state.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        # We disable reverse state for now (by setting it to a non-existent state)
        self.REVERSE_STATE = 5

        self.GOAL_STATE = 3
        self.NUM_STATES = self.GOAL_STATE + 1

        self.current_state = 0
        self.goal_reached = False

        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state and returns the initial observation and info.

        Returns:
            observation (int): The initial observation (always 0).
            info (dict): An empty dictionary.
        """
        super().reset(seed=seed)
        self.goal_reached = False
        self.current_state = 0
        return 0, {}

    def step(self, action):
        """
        Executes a single time step within the environment.

        Args:
            action (int): The action taken (0 or 1).

        Returns:
            observation (int): The next observation (always 0).
            reward (int): A reward, here fixed to -1.
            terminated (bool): True when the goal state is reached.
            truncated (bool): Always False, since no truncation logic is present.
            info (dict): Additional debug information (empty here).
        """
        step = -1 if action == 0 else 1

        if self.current_state == self.REVERSE_STATE:
            step = -step

        self.current_state += step

        self.current_state = max(0, self.current_state)
        self.current_state = self.current_state % (self.NUM_STATES)

        observation = 0
        reward = -1
        terminated = self.current_state >= self.GOAL_STATE
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        """
        Renders a simple textual representation of the corridor.
        """
        corridor = ""
        for i in range(self.NUM_STATES):
            marker = "x" if self.current_state == i else " "
            if i == self.REVERSE_STATE:
                corridor += "{" + marker + "}"
            else:
                corridor += "[" + marker + "]"

        print(corridor)
