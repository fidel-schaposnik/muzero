import gym
import tensorflow as tf

from muzero.exceptions import MuProverImplementationError, MuProverEnvironmentError

# For type annotations
from typing import List, Tuple

from muzero.muprover_types import State, Value, Action


class Environment:
    """
    A class for environments with which MuZero interacts (restricted to 1-player games).
    Sub-class this to implement your own environments, implementing:

    Methods following the OpenAI API (see https://github.com/openai/gym/blob/master/gym/core.py for more details):
        - step
        - reset

    Other methods:
        - is_legal_action
    """

    def __init__(self, action_space_size: int) -> None:
        self.action_space_size: int = action_space_size

    def step(self, action: Action) -> Tuple[State, Value, bool, dict]:
        """
        Run one step of the environment's dynamics. When end of episode is reached, you are responsible for
        resetting this environment's state.

        Returns:
            - observation (tf.Tensor): agent's observation of the current environment
            - reward (float): amount of reward returned after previous action
            - done (bool): whether the episode has ended, in which case further step() calls return undefined results
            - info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise MuProverImplementationError('step', 'Environment')

    def reset(self) -> State:
        """
        Resets the environment to an initial state and returns an initial observation.
        """
        raise MuProverImplementationError('reset', 'Environment')

    def is_legal_action(self, action: Action) -> bool:
        """
        Returns True if the action is legal in the current state, False otherwise.
        """
        raise MuProverImplementationError('is_legal_action', 'Environment')

    def legal_actions(self) -> List[Action]:
        """
        Returns a list of all actions that can be performed in the current state.
        """
        return list(filter(self.is_legal_action, range(self.action_space_size)))


class OpenAIEnvironment(Environment):
    def __init__(self, gym_id: str) -> None:
        self.env = gym.make(gym_id)
        if type(self.env.action_space) != gym.spaces.discrete.Discrete:
            raise MuProverEnvironmentError(message='only environments with discrete action spaces are supported')

        super().__init__(action_space_size=self.env.action_space.n)

    def step(self, action: Action) -> Tuple[State, Value, bool, dict]:
        observation, reward, done, info = self.env.step(action)
        return tf.constant(observation, dtype=tf.float32), Value(reward), done, info

    def reset(self) -> State:
        return tf.constant(self.env.reset(), dtype=tf.float32)

    def is_legal_action(self, action: Action) -> bool:
        return action in range(0, self.action_space_size)
