import random
import time
import argparse
import numpy as np
import tensorflow as tf
from math import exp

from game import *
from exceptions import *
from environment import RemoteEnvironment, Player
from replay_buffer import RemoteReplayBuffer
from network import RemoteNetwork
from utils import MinMaxStats, load_game


class Agent:
    def play_game(self, environment):
        """
        Plays a game in the Environment.
        """
        start = time.time()
        game = Game(environment=environment)
        while not game.terminal():
            action = self.make_move(game)
            # print('Player {} performs {}'.format(game.to_play(), action))
            game.apply(action)
        end = time.time()
        print('Finished playing a game in {:.2f} seconds, {:.2f} seconds per move!'.format(end-start, (end-start)/len(game.history)))
        return game.history

    def make_move(self, game):
        """
        Choose a move to play in the Game.
        """
        raise ImplementationError('make_move', 'Agent')


class RandomAgent(Agent):
    """
    Completely random agent, for testing purposes.
    """
    def make_move(self, game):
        return random.choice(game.legal_actions())


class Node:
    def __init__(self, prior=1.0, parent=None):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        self.parent = parent

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def print(self, _prefix='', name='Root', _last=True):
        print(_prefix, '`- ' if _last else '|- ', '{}: value={}; reward={}'.format(name, float(self.value()), float(self.reward)), sep="")
        _prefix += '   ' if _last else '|  '
        child_count = len(self.children)
        for i, (action, child) in enumerate(self.children.items()):
            _last = i == (child_count - 1)
            child.print(_prefix, action, _last)


class NetworkAgent(Agent):
    """
    Agent greedily choosing the best action according to a network's policy outputs.
    This is essentially like MCTSAgent with num_simulations = 0.
    """
    def __init__(self, network):
        self.network = network

    def make_move(self, game):
        observation = np.array([game.make_image()], dtype=np.float32)
        policy_logits = self.network.initial_inference(observation).split_batch()[0].policy_logits
        # print(policy_logits)
        _, action = max([(policy_logits[action], action) for action in game.legal_actions()])
        return action

class MCTSAgent(Agent):
    """
    Use Monte-Carlo Tree-Search to select moves.
    """
    def __init__(self, mcts_config, network):
        self.config = mcts_config
        self.network = network

    @staticmethod
    def expand_node(node, to_play, actions, network_output):
        node.to_play = to_play
        node.hidden_state = network_output.hidden_state
        node.reward = network_output.reward
        policy = {a: exp(network_output.policy_logits[a]) for a in actions}
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            node.children[action] = Node(prior=p/policy_sum, parent=node)

    @staticmethod
    def softmax_sample(distribution, temperature):
        if temperature == 0.0:
            return max(distribution)
        else:
            weights = np.array([count ** (1 / temperature) for count, action in distribution])
            weights /= sum(weights)
            return random.choices(distribution, weights=weights)[0]

    def add_exploration_noise(self, node):
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
        frac = self.config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    def select_leaf(self, node, num_moves, min_max_stats):
        action = None
        while node.expanded():
            action, node = self.select_child(node, min_max_stats)
            num_moves += 1
        return action, node, num_moves

    def select_child(self, node, min_max_stats):
        # print({action: float(self.ucb_score(child, min_max_stats)) for action, child in node.children.items()})
        _, action, child = max(
            (self.ucb_score(child, min_max_stats), action, child) for action, child in node.children.items())
        # print('Going down {}'.format(action))
        # input('Press any key to continue...')
        return action, child

    def ucb_score(self, node, min_max_stats):
        exploration_score = node.prior * self.config.exploration_function(node.parent.visit_count, node.visit_count)
        if node.visit_count > 0:
            exploitation_score = node.reward + self.config.game_config.discount * min_max_stats.normalize(node.value())
        else:
            exploitation_score = 0
        return exploration_score + exploitation_score

    def backpropagate(self, node, value, to_play, min_max_stats):
        while node:
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.value())
            value = node.reward + self.config.game_config.discount * value  # This probably needs to take into account to_play!
            node = node.parent

    def run_mcts(self, root, num_moves):
        min_max_stats = MinMaxStats(self.config.known_bounds)

        for _ in range(self.config.num_simulations):
            # root.print()
            action, leaf, cur_moves = self.select_leaf(root, num_moves, min_max_stats)
            to_play = Player(cur_moves % self.config.game_config.num_players)

            batch_hidden_state = tf.expand_dims(leaf.parent.hidden_state, axis=0)
            network_output = self.network.recurrent_inference(batch_hidden_state, [action]).split_batch()[0]
            self.expand_node(node=leaf, to_play=to_play, actions=self.config.game_config.action_space,
                             network_output=network_output)
            self.backpropagate(leaf, network_output.value, to_play, min_max_stats)

    def select_action(self, node, num_moves):
        visit_counts = [(child.visit_count, action) for action, child in node.children.items()]
        t = self.config.visit_softmax_temperature_fn(num_moves=num_moves, training_steps=self.network.training_steps())
        _, action = self.softmax_sample(visit_counts, t)
        return action

    def make_move(self, game):
        root = Node()
        observation = np.array([game.make_image()], dtype=np.float32)
        self.expand_node(node=root, to_play=game.to_play(), actions=game.legal_actions(),
                         network_output=self.network.initial_inference(observation).split_batch()[0])
        self.add_exploration_noise(root)

        self.run_mcts(root, len(game.history))
        game.store_search_statistics(root)
        return self.select_action(root, len(game.history))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuZero MCTS Agent')
    parser.add_argument('--game', type=str, required=True,
                        help='One of the games implemented in the games/ directory')
    parser.add_argument('--replay_buffer', type=str, required=True,
                        help='IP:Port for gRPC communication with a replay buffer server')
    parser.add_argument('--environment', type=str, required=True,
                        help='IP:Port for gRPC communication with an environment server')
    parser.add_argument('--network', type=str, required=True,
                        help='IP:Port for gRPC communication with a network server')
    parser.add_argument('--num_games', type=int, default=int(1e10),
                        help='Number of games to play (defaults to infinity)')
    args = parser.parse_args()

    config = load_game(args.game, parser)

    remote_replay_buffer = RemoteReplayBuffer(ip_port=args.replay_buffer)

    remote_network = RemoteNetwork(network_config=config.network_config,
                                   ip_port=args.network)

    agent = MCTSAgent(config.mcts_config, remote_network)

    for _ in range(args.num_games):
        remote_environment = RemoteEnvironment(game_config=config.game_config,
                                               ip_port=args.environment)
        game_history = agent.play_game(remote_environment)
        remote_replay_buffer.save_history(game_history)
