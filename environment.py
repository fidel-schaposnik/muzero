import gym
import tensorflow as tf

from exceptions import MuZeroImplementationError, MuZeroEnvironmentError

# For type annotations
from typing import List, Tuple

from muzero_types import State, Observation, Player, Action, Value


class Environment:
    """
    A class for environments with which MuZero interacts.
    Sub-class this to implement your own environments, implementing:
        - step
        - reset
    """

    def __init__(self, action_space_size: int, num_players: int) -> None:
        self.action_space_size: int = action_space_size
        self.num_players: int = num_players

    def step(self, action: Action) -> Tuple[State, Value, bool, dict]:
        """
        Run one step of the environment's dynamics. When end of episode is reached, you are responsible for
        resetting this environment's state.

        Returns:
            - state (observation, to_play, legal_actions): agent's observation of the current environment
            - reward (float): amount of reward returned for the previous action
            - done (bool): whether the episode has ended, in which case further step() calls return undefined results
            - info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise MuZeroImplementationError('step', 'Environment')

    def reset(self) -> State:
        """
        Resets the environment to an initial state and returns an initial state.
        """
        raise MuZeroImplementationError('reset', 'Environment')


class OpenAIEnvironment(Environment):
    """A class for single-player OpenAI environments with a Discrete action space."""

    def __init__(self, gym_id: str) -> None:
        self.env = gym.make(gym_id)
        if type(self.env.action_space) != gym.spaces.discrete.Discrete:
            raise MuZeroEnvironmentError(message='only environments with discrete action spaces are supported')

        super().__init__(action_space_size=self.env.action_space.n, num_players=1)

    def _legal_actions(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def step(self, action: Action) -> Tuple[State, Value, bool, dict]:
        observation, reward, done, info = self.env.step(action)
        state = State(Observation(tf.constant(observation, dtype=tf.float32)), Player(0), self._legal_actions())
        return state, Value(reward), done, info

    def reset(self) -> State:
        return State(Observation(tf.constant(self.env.reset(), dtype=tf.float32)), Player(0), self._legal_actions())
