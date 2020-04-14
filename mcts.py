from utils import *
from math import sqrt, log, exp
import random, time


class Node:
    def __init__(self, parent=None, gets_reward=None, prior=0.0):
        self.parent = parent
        self.gets_reward = gets_reward
        self.prior = prior
        self.children = {}
        self.reward_dict = {}
        self.hidden_state = None
        self.num_simulations = 0
        self.value_dict_sum = {}
        self.value = 0

    def expanded(self):
        return len(self.children) > 0

    def update_value(self, discount):
        self.value = self.reward_dict.get(self.gets_reward, 0) + discount * self.value_dict_sum.get(self.gets_reward, 0) / self.num_simulations
        return self.value


def select_leaf(config, node, min_max_stats):
    while node.expanded():
        action, node = select_child(config, node, min_max_stats)
    return action, node


def select_child(config, node, min_max_stats):
    # if not node.parent:
    #     for action, child in node.children.items():
    #         print('{}-{}: ({:.2f}, {:.2f})'.format(action, child.num_simulations, min_max_stats.normalize(child.value), child.prior * config.exploration_function(node.num_simulations, child.num_simulations)), end=', ')
    #     print()
    #     input('Press to continue')
    _, action, child = max((ucb_score(config, child, min_max_stats), action, child) for action, child in node.children.items())
    # print('Going down {}'.format(action))
    return action, child


def ucb_score(config, node, min_max_stats):
    return min_max_stats.normalize(node.value) + node.prior * config.exploration_function(node.parent.num_simulations, node.num_simulations)


def expand_node(node, to_play, actions, network_output):
    policy = {action: exp(network_output.policy_logits[action]) for action in actions}
    policy_sum = sum(policy.values())
    node.children = {action: Node(parent=node, gets_reward=to_play, prior=p/policy_sum) for action, p in policy.items()}
    node.reward_dict = network_output.reward
    node.hidden_state = network_output.hidden_state


def backpropagate(node, value_dict, discount, min_max_stats):
    while node:
        for player, value in value_dict.items():
            node.value_dict_sum[player] = node.value_dict_sum.setdefault(player, 0) + value
        node.num_simulations += 1
        min_max_stats.update(node.update_value(discount))

        for player, value in value_dict.items():
            value_dict[player] *= discount
        for player, reward in node.reward_dict.items():
            value_dict[player] = reward + value_dict.setdefault(player, 0)
        node = node.parent


def batch_mcts(config, batch_root, network):
    min_max_stats = MinMaxStats(config.known_bounds)

    batch_hidden_state = np.empty(network.hidden_state_shape(len(batch_root)), dtype=np.float32)

    # TRANSVERSAL_TIME = EXPANSION_TIME = 0
    for _ in range(config.num_simulations):
        batch_leaf, batch_last_action = [], []
        # start = time.time()
        for i, root in enumerate(batch_root):
            last_action, leaf = select_leaf(config, root, min_max_stats)
            batch_last_action.append(last_action)
            batch_leaf.append(leaf)
            batch_hidden_state[i] = leaf.parent.hidden_state
        # end = time.time()
        # TRANSVERSAL_TIME += end - start

        batch_network_output = network.recurrent_inference(batch_hidden_state, batch_last_action)

        # start = time.time()
        for leaf, network_output in zip(batch_leaf, batch_network_output.split_batch()):
            expand_node(leaf, network_output.to_play, config.action_space, network_output)
            backpropagate(leaf, network_output.value, config.discount, min_max_stats)
        # end = time.time()
        # EXPANSION_TIME += end - start
    # return TRANSVERSAL_TIME, EXPANSION_TIME


def softmax_sample(distribution, temperature, training):
    if not training or temperature == 0.0:
        return max(distribution)
    else:
        weights = np.array([count**(1/temperature) for count, action in distribution])
        weights /= sum(weights)
        return random.choices(distribution, weights=weights)[0]


def select_action(config, node, num_moves, num_steps, training):
    visit_counts = [(child.num_simulations, action) for action, child in node.children.items()]
    temperature = config.visit_softmax_temperature_fn(num_moves=num_moves, num_steps=num_steps)
    _, action = softmax_sample(visit_counts, temperature, training)
    return action


def add_exploration_noise(config, node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def batch_make_move(config, network, games, training):
    batch_root = [Node() for _ in games]
    batch_current_observation = np.array([game.make_image() for game in games])
    batch_initial_inference = network.initial_inference(batch_current_observation)

    # start = time.time()
    for root, game, initial_inference in zip(batch_root, games, batch_initial_inference.split_batch()):
        expand_node(root, game.to_play(), game.legal_actions(), initial_inference)
        add_exploration_noise(config, root)
    # end = time.time()
    # PRE_TIME = end-start

    # TRANSVERSAL_TIME, EXPANSION_TIME =
    batch_mcts(config, batch_root, network)

    # start = time.time()
    for root, game in zip(batch_root, games):
        game.store_search_statistics(root)
        action = select_action(config, root, num_moves=len(game.history), num_steps=network.training_steps(), training=training)
        game.apply(action)
    # end = time.time()
    # POST_TIME = end-start
    #
    # return TRANSVERSAL_TIME, EXPANSION_TIME, PRE_TIME, POST_TIME
