from environment import *
import tensorflow as tf


class MuZeroConfig:
    def __init__(self,
                 name,
                 weight_decay,
                 window_size,
                 batch_size,
                 num_unroll_steps,
                 td_steps,
                 training_steps,
                 checkpoint_interval,
                 learning_rate,
                 num_simulations,
                 known_bounds,
                 discount,
                 freezing_moves,
                 root_dirichlet_alpha,
                 root_exploration_noise,
                 max_moves,
                 game_class,
                 network_class,
                 action_space_size,
                 **game_params):

        def visit_softmax_temperature(num_moves, num_steps):
            if num_moves < freezing_moves:
                return 1.0
            else:
                return 0.0  # Play according to the max.

        # Game parameters
        self.name = name
        self.game_class = game_class
        self.network_class = network_class
        self.action_space_size = action_space_size
        self.action_space = [Action(i) for i in range(action_space_size)]
        self.game_params = game_params

        # MCTS parameters
        self.num_simulations = num_simulations
        self.known_bounds = known_bounds
        self.discount = discount
        self.visit_softmax_temperature_fn = visit_softmax_temperature
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_fraction = root_exploration_noise
        self.max_moves = max_moves

        # Training parameters
        if 'scalar_support_size' in game_params.keys():
            self.value_loss = tf.keras.losses.categorical_crossentropy
            self.reward_loss = tf.keras.losses.categorical_crossentropy
        else:
            self.value_loss = tf.keras.losses.mean_squared_error
            self.reward_loss = tf.keras.losses.mean_squared_error

        self.weight_decay = weight_decay
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps
        self.training_steps = training_steps
        self.checkpoint_interval = checkpoint_interval
        self.learning_rate = learning_rate

    def new_game(self):
        return self.game_class(**self.game_params)

    def make_uniform_network(self):
        return self.network_class(**self.game_params)
