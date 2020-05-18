from math import log, sqrt
from environment import Action

class ReplayBufferConfig:
    def __init__(self,
                 window_size
                 ):
        self.window_size = window_size


class GameConfig:
    def __init__(self,
                 name,
                 environment_class,
                 environment_parameters,
                 num_players,
                 action_space_size,
                 discount):
        self.name = name
        self.environment_class = environment_class
        self.environment_parameters = environment_parameters
        self.num_players = num_players
        self.action_space_size = action_space_size
        self.discount = discount

        self.action_space = [Action(index) for index in range(action_space_size)]


class TrainingConfig:
    def __init__(self,
                 game_config,
                 batch_size,
                 num_unroll_steps,
                 td_steps,
                 optimizer,
                 training_steps,
                 checkpoint_interval,
                 value_loss_decay,
                 value_loss,
                 reward_loss_decay,
                 reward_loss,
                 regularization_decay
                 ):
        self.game_config = game_config
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps
        self.optimizer = optimizer
        self.training_steps = training_steps
        self.checkpoint_interval = checkpoint_interval
        self.value_loss_decay = value_loss_decay
        self.value_loss = value_loss
        self.reward_loss_decay = reward_loss_decay
        self.reward_loss = reward_loss
        self.regularization_decay = regularization_decay


class MCTSConfig:
    def __init__(self,
                 game_config,
                 max_moves,
                 root_dirichlet_alpha,
                 root_exploration_fraction,
                 known_bounds,
                 num_simulations,
                 freezing_moves
                 ):
        self.game_config = game_config
        self.max_moves = max_moves
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_fraction = root_exploration_fraction
        self.known_bounds = known_bounds
        self.num_simulations = num_simulations
        self.freezing_moves = freezing_moves

    def visit_softmax_temperature_fn(self, num_moves, training_steps):
        if num_moves < self.freezing_moves:
            return 1.0
        else:
            return 0.0

    def exploration_function(self, parent_visit_count, child_visit_count):
        return sqrt(2 * parent_visit_count) / (child_visit_count + 1)


class NetworkConfig:
    def __init__(self, network_class, state_action_encoder, network_parameters):
        self.network_class = network_class
        self.state_action_encoder = state_action_encoder
        self.network_parameters = network_parameters

    def make_uniform_network(self):
        return self.network_class(state_action_encoder=self.state_action_encoder, **self.network_parameters)


class MuZeroConfig:
    def __init__(self, game_config, replay_buffer_config, training_config, mcts_config, network_config):
        self.game_config = game_config
        self.replay_buffer_config = replay_buffer_config
        self.training_config = training_config
        self.mcts_config = mcts_config
        self.network_config = network_config
