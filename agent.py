import random
import time
import tensorflow as tf
import numpy as np
from math import isnan

from game import Game
from exceptions import MuZeroImplementationError
from utils import random_id, MinMaxStats

# For type annotations
from typing import Optional, Dict, List, Tuple, Union

from muzero_types import Player, Action, Policy, Value, Observation, ObservationBatch, ActionBatch
from game import GameHistory
from environment import Environment
from network import Network, NetworkOutput
from network_services import RemoteNetwork
from config import MuZeroConfig


class Agent:
    """
    An agent playing a game in an Environment.
    """

    def __init__(self, config: MuZeroConfig, agent_id: Optional[str] = None) -> None:
        self.config: MuZeroConfig = config
        self.agent_id: str = agent_id if agent_id else 'agent_'+random_id()

    def play_game(self, environment: Environment) -> GameHistory:
        start = time.time()
        game = Game(environment=environment)
        while not game.terminal():
            # game.environment.env.render()
            action = self.make_move(game)
            # print(f'Performing {action}')
            game.apply(action)
        end = time.time()

        game.history.metadata['agent_id'] = self.agent_id
        game.history.metadata['timing'] = end-start
        game.history.metadata.update(self.fill_metadata())
        return game.history

    def make_move(self, game: Game) -> Action:
        """
        Choose a move to play in the Game.
        """
        raise MuZeroImplementationError('make_move', 'Agent')

    def fill_metadata(self) -> Dict[str, str]:
        """
        Sub-classed agents can use this callback to add further metadata to saved games.
        """
        return {}


class RandomAgent(Agent):
    """
    Completely random agent, for testing purposes.
    """

    def make_move(self, game: Game) -> Action:
        legal_actions = game.legal_actions()
        policy = np.zeros(game.environment.action_space_size)
        policy[legal_actions] = 1 / len(legal_actions)
        a, b = self.config.value_config.known_bounds.endpoints() if self.config.value_config.known_bounds else (0, 1)
        value = a + (a - b)*random.random()
        game.store_search_statistics(Value(value), Policy(tf.constant(policy)))
        return random.choice(legal_actions)


class NetworkAgent(Agent):
    """
    Agent choosing the next action according to a network's policy outputs.
    This is roughly like MCTSAgent with num_simulations = 0.
    """

    def __init__(self, config: MuZeroConfig, network: Network, temperature: float = 0.0, debug: bool = False) -> None:
        super().__init__(config=config)
        self.network: Network = network
        self.temperature: float = temperature
        self.debug: bool = debug

    def make_move(self, game: Game) -> Action:
        observation_batch = ObservationBatch(tf.expand_dims(game.history.make_image(), axis=0))
        batch_network_output = self.network.initial_inference(observation_batch)
        network_output = batch_network_output.split_batch()[0]

        legal_actions = game.legal_actions()
        policy = network_output.masked_policy(legal_actions)
        game.store_search_statistics(network_output.value, Policy(tf.constant(policy)))

        if self.temperature == 0:
            _, action = max(zip(policy[legal_actions], legal_actions))
            return action
        else:
            weights = policy[legal_actions] ** (1 / self.temperature)
            return random.choices(legal_actions, weights=weights, k=1)[0]


class Node:
    def __init__(self, parent: Optional['Node'] = None, prior: float = 1.0) -> None:
        self.parent: Optional[Node] = parent
        self.prior: float = prior
        self.children: Dict[Action, Node] = {}
        self.hidden_state: Optional[Observation] = None
        self.reward: Optional[Value] = None
        self.to_play: Optional[Player] = None

        self.value_sum: Value = Value(0.0)
        self.visit_count: int = 0
        self.value: Value = Value(float('nan'))

    def expanded(self) -> bool:
        return len(self.children) > 0

    def update_value(self, value: Value) -> Value:
        self.value_sum += value
        self.visit_count += 1
        self.value = Value(self.value_sum / self.visit_count)
        return self.value

    def print(self, _prefix: str = '', name: str = 'Root', _last: bool = True) -> None:
        print(_prefix, '`- ' if _last else '|- ',
              f'{name}-{self.visit_count}: prior={self.prior:.2f}; value={self.value:.4f}', sep="")
        _prefix += '   ' if _last else '|  '
        child_count = len(self.children)
        for i, (action, child) in enumerate(self.children.items()):
            _last = i == (child_count - 1)
            child.print(_prefix, action, _last)


class MCTSAgent(Agent):
    """
    Use Monte-Carlo Tree-Search to select moves.
    """
    def __init__(self,
                 config: MuZeroConfig,
                 network: Union[Network, RemoteNetwork],
                 agent_id: Optional[str] = None,
                 debug: bool = False
                 ) -> None:
        super().__init__(config=config, agent_id=agent_id)
        self.config: MuZeroConfig = config
        self.network: Network = network
        self.debug: bool = debug

        self.effective_discount: float = self.config.game_config.discount
        if config.game_config.num_players == 2:
            self.effective_discount *= -1

    @staticmethod
    def expand_node(node: Node, actions: List[Action], network_output: NetworkOutput) -> None:
        node.hidden_state = network_output.hidden_state
        node.reward = network_output.reward
        policy = network_output.masked_policy(actions)
        for action, p in zip(actions, policy):
            node.children[action] = Node(prior=p, parent=node)

    @staticmethod
    def softmax_sample(distribution: List[Tuple[int, Action]], temperature: float) -> Tuple[int, Action]:
        if temperature == 0.0:
            return max(distribution)
        else:
            weights = [count ** (1 / temperature) for count, action in distribution]
            return random.choices(distribution, weights=weights, k=1)[0]

    def add_exploration_noise(self, node: Node) -> None:
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.mcts_config.root_dirichlet_alpha] * len(actions))
        frac = self.config.mcts_config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    def select_leaf(self, node: Node, min_max_stats: MinMaxStats) -> Tuple[Action, Node]:
        action = Action(-1)
        while node.expanded():
            action, node = self.select_child(node, min_max_stats)
        return action, node

    def select_child(self, node: Node, min_max_stats: MinMaxStats) -> Tuple[Action, Node]:
        _, action, child = max((self.ucb_score(child, min_max_stats), a, child) for a, child in node.children.items())
        return action, child

    def ucb_score(self, node: Node, min_max_stats: MinMaxStats) -> float:
        exploitation_score = self.config.mcts_config.default_value if isnan(node.value) else node.reward + self.effective_discount * node.value
        exploration_score = node.prior * self.config.exploration_function(node.parent.visit_count, node.visit_count)
        return min_max_stats.normalize(exploitation_score) + exploration_score

    def backpropagate(self, node: Node, value: Value, min_max_stats: MinMaxStats) -> None:
        while node is not None:
            min_max_stats.update(node.update_value(value))
            value = node.reward + self.effective_discount * value if node.reward is not None else Value(float('nan'))
            node = node.parent

    def run_mcts(self, root: Node, min_max_stats: MinMaxStats) -> None:
        for _ in range(self.config.mcts_config.num_simulations):
            if self.debug:
                root.print()
            action, leaf = self.select_leaf(root, min_max_stats)

            batch_hidden_state = ObservationBatch(tf.expand_dims(leaf.parent.hidden_state, axis=0))
            batch_action = ActionBatch(tf.constant([action]))
            batch_network_output = self.network.recurrent_inference(batch_hidden_state, batch_action)
            network_output = batch_network_output.split_batch()[0]
            self.expand_node(node=leaf,
                             actions=self.config.action_space(),
                             network_output=network_output)
            self.backpropagate(leaf, network_output.value, min_max_stats)

    def select_action(self, node: Node, num_moves: int) -> Action:
        visit_counts = [(child.visit_count, action) for action, child in node.children.items()]
        t = self.config.visit_softmax_temperature_fn(num_moves=num_moves, training_steps=self.network.training_steps())
        _, action = self.softmax_sample(visit_counts, t)
        return action

    def make_move(self, game: Game) -> Action:
        root = Node()
        min_max_stats = MinMaxStats(known_bounds=self.config.value_config.known_bounds)
        observation = ObservationBatch(tf.expand_dims(game.history.make_image(), axis=0))
        self.expand_node(node=root,
                         actions=game.legal_actions(),
                         network_output=self.network.initial_inference(observation).split_batch()[0])
        self.add_exploration_noise(root)
        self.run_mcts(root, min_max_stats)

        action_space = self.config.action_space()
        policy = [root.children[a].visit_count / root.visit_count if a in root.children else 0 for a in action_space]
        game.store_search_statistics(root.value, Policy(tf.constant(policy)))
        return self.select_action(root, len(game.history))

    def fill_metadata(self) -> Dict[str, str]:
        return {'network_id': str(self.network.training_steps())}
