# For type annotations
from typing import List, Dict, Optional, Any

from muzero_types import State, Observation, Player, Action, Value, Policy, ValueBatch, PolicyBatch, ActionBatch
from environment import Environment


class GameHistory:
    """
    Book-keeping class for completed games.
    Restricted to 1-player games with no intermediate rewards for MuProver.
    """

    def __init__(self) -> None:
        self.observations: List[Observation] = []
        self.to_plays: List[Player] = []
        self.actions: List[Action] = []
        self.rewards: List[Value] = []
        self.root_values: List[Value] = []
        self.policies: List[Policy] = []
        self.metadata: Dict[str, Any] = {}

        # The following are only filled once within a replay buffer
        self.extended_actions: Optional[ActionBatch] = None
        self.target_rewards: Optional[ValueBatch] = None
        self.target_values: Optional[ValueBatch] = None
        self.target_policies: Optional[PolicyBatch] = None
        self.total_value: Value = Value(float('nan'))

    def make_image(self, index: int = -1) -> Observation:
        """
        TODO: If necessary, stack multiple states to create an observation.
        """
        return self.observations[index]

    def __repr__(self) -> str:
        return 'Game({})'.format(', '.join(map(str, self.actions)))

    def __len__(self) -> int:
        return len(self.actions)


class Game:
    """
    A class to record episodes of interaction with an Environment.
    """

    def __init__(self, environment: Environment) -> None:
        self.environment: Environment = environment
        self.history: GameHistory = GameHistory()

        self.state: State = self.environment.reset()
        self.history.observations.append(self.state.observation)
        self.history.to_plays.append(self.state.to_play)
        self.ended: bool = False

    def to_play(self) -> Player:
        return self.state.to_play

    def legal_actions(self) -> List[Action]:
        return self.state.legal_actions

    def terminal(self) -> bool:
        return self.ended

    def apply(self, action: Action) -> None:
        self.state, reward, self.ended, info = self.environment.step(action)
        self.history.observations.append(self.state.observation)
        self.history.to_plays.append(self.state.to_play)
        self.history.actions.append(action)
        self.history.rewards.append(reward)

    def store_search_statistics(self, value: Value, policy: Policy) -> None:
        self.history.root_values.append(value)
        self.history.policies.append(policy)
