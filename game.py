# For type annotations
from typing import List, Dict, Optional, Any

from muzero.muprover_types import State, Observation, Value, Policy, ValueBatch, PolicyBatch, ActionBatch, Action
from muzero.environment import Environment


class GameHistory:
    """
    Book-keeping class for completed games.
    Restricted to 1-player games with no intermediate rewards for MuProver.
    """

    def __init__(self) -> None:
        self.states: List[State] = []
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
        return Observation(self.states[index])

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
        self.history.states.append(self.environment.reset())
        self.ended: bool = False

    def legal_actions(self) -> List[Action]:
        return self.environment.legal_actions()

    def terminal(self) -> bool:
        return self.ended

    def apply(self, action: Action) -> None:
        state, reward, self.ended, info = self.environment.step(action)
        self.history.states.append(state)
        self.history.actions.append(action)
        self.history.rewards.append(reward)

    def store_search_statistics(self, policy: Policy, value: Value) -> None:
        self.history.root_values.append(value)
        self.history.policies.append(policy)
