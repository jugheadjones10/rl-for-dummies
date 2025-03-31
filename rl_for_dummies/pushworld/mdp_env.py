import abc
from typing import Generic, Tuple, TypeVar

import gymnasium as gym
import numpy as np

ObsType = TypeVar("ObsType")


class MetaEpisodicEnv(abc.ABC, Generic[ObsType]):
    @property
    @abc.abstractmethod
    def max_episode_len(self) -> int:
        """
        Return the maximum episode length.
        """
        pass

    @abc.abstractmethod
    def new_env(self) -> None:
        """
        Reset the environment's structure by resampling
        the state transition probabilities and/or reward function
        from a prior distribution.

        Returns:
            None
        """
        pass

    @abc.abstractmethod
    def reset(self) -> ObsType:
        """
        Resets the environment's state to some designated initial state.
        This is distinct from resetting the environment's structure
            via self.new_env().

        Returns:
            initial observation.
        """
        pass

    @abc.abstractmethod
    def step(
        self, action: int, auto_reset: bool = True
    ) -> Tuple[ObsType, float, bool, dict]:
        """
        Step the env.

        Args:
            action: integer action indicating which action to take
            auto_reset: whether or not to automatically reset the environment
                on done. if true, next observation will be given by self.reset()

        Returns:
            next observation, reward, and done flat
        """
        pass


class MDPEnv(MetaEpisodicEnv):
    """
    Tabular MDP env with support for resettable MDP params (new meta-episode),
    in addition to the usual reset (new episode).
    """

    def __init__(self, num_states, num_actions, max_episode_length=10):
        # structural
        self._num_states = num_states
        self._num_actions = num_actions
        self._max_ep_length = max_episode_length
        self._action_space = gym.spaces.Discrete(num_actions)

        # per-environment-sample quantities.
        self._reward_means = None
        self._state_transition_probabilities = None
        self.new_env()

        # mdp state.
        self._ep_steps_so_far = 0
        self._state = 0

    @property
    def action_space(self):
        return self._action_space

    @property
    def max_episode_len(self):
        return self._max_ep_length

    @property
    def num_actions(self):
        """Get self._num_actions."""
        return self._num_actions

    @property
    def num_states(self):
        """Get self._num_states."""
        return self._num_states

    def _new_reward_means(self):
        self._reward_means = np.random.normal(
            loc=1.0, scale=1.0, size=(self._num_states, self._num_actions)
        )

    def _new_state_transition_dynamics(self):
        p_aijs = []
        for a in range(self._num_actions):
            dirichlet_samples_ij = np.random.dirichlet(
                alpha=np.ones(dtype=np.float32, shape=(self._num_states,)),
                size=(self._num_states,),
            )
            p_aijs.append(dirichlet_samples_ij)
        self._state_transition_probabilities = np.stack(p_aijs, axis=0)

    def new_env(self) -> None:
        """
        Sample a new MDP from the distribution over MDPs.

        Returns:
            None
        """
        self._new_reward_means()
        self._new_state_transition_dynamics()
        self._state = 0

    def reset(self) -> Tuple[int, dict]:
        """
        Reset the environment.

        Returns:
            initial state.
        """
        self._ep_steps_so_far = 0
        self._state = 0
        return self._state, {}

    def step(self, action, auto_reset=True) -> Tuple[int, float, bool, bool, dict]:
        """
        Take action in the MDP, and observe next state, reward, done, etc.

        Args:
            action: action corresponding to an arm index.
            auto_reset: auto reset. if true, new_state will be from self.reset()

        Returns:
            new_state, reward, done, info.
        """
        self._ep_steps_so_far += 1
        t = self._ep_steps_so_far

        s_t = self._state
        a_t = action

        s_tp1 = np.random.choice(
            a=self._num_states, p=self._state_transition_probabilities[a_t, s_t]
        )
        self._state = s_tp1

        r_t = np.random.normal(loc=self._reward_means[s_t, a_t], scale=1.0)

        done_t = False if t < self._max_ep_length else True
        if done_t and auto_reset:
            s_tp1 = self.reset()

        # We add False truncated to the return tuple to adhere to Gymnasium API
        return s_tp1, r_t, done_t, False, {}
