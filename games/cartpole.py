import gym

from config import *
from network import *


def make_config():
    game_config = GameConfig(name='CartPole',
                             environment_class=CartPoleEnvironment,
                             environment_parameters={},
                             action_space_size=2,
                             num_players=1,
                             discount=0.99
                             )

    replay_buffer_config = ReplayBufferConfig(window_size=int(1e5))

    mcts_config = MCTSConfig(max_moves=500,
                             root_dirichlet_alpha=1.0,
                             root_exploration_fraction=0.25,
                             known_bounds=None,
                             num_simulations=16,
                             game_config=game_config
                             )

    network_config = NetworkConfig(network_class=CartPoleNetwork,
                                   state_action_encoder=BinaryPlaneEncoder(),
                                   network_parameters={
                                       'num_layers': 2, 'num_units': 64,  # Fully connected network parameters
                                       'policy_filters': 32, 'policy_kernel_size': (3, 3),  # Policy head parameters
                                       'value_filters': 32, 'value_kernel_size': (3, 3),  # Value head parameters
                                       'reward_filters': 32, 'reward_kernel_size': (3, 3),  # Reward head parameters
                                       'hidden_size': 32, 'scalar_activation': 'relu'  # Parameters shared by value and reward heads
                                   }
                                   )

    training_config = TrainingConfig(game_config=game_config,
                                     batch_size=2048,
                                     num_unroll_steps=5,
                                     td_steps=10,
                                     optimizer=tf.keras.optimizers.Adam(lr=.001),
                                     training_steps=int(1e5),
                                     checkpoint_interval=int(5e2),
                                     value_loss_decay=1.0,
                                     value_loss=tf.keras.losses.mean_squared_error,
                                     reward_loss_decay=1.0,
                                     reward_loss=tf.keras.losses.mean_squared_error,
                                     regularization_decay=1e-3
                                     )

    return MuZeroConfig(game_config=game_config,
                        replay_buffer_config=replay_buffer_config,
                        mcts_config=mcts_config,
                        training_config=training_config,
                        network_config=network_config)


class CartPoleEnvironment(Environment):
    """
    The environment class of cart-pole.
    """

    def __init__(self, **kwargs):  # kwargs collects arguments not used here (network parameters)
        super().__init__(action_space_size=2, num_players=1)
        self.env = gym.make('CartPole-v1')
        self.state = self.env.reset()
        self.ended = False
        self.cumulative_reward = 0.0
        self.max_moves = self.env._max_episode_steps

    def is_legal_action(self, action):
        return True

    def to_play(self):
        return Player(0)

    def terminal(self):
        return self.ended

    def outcome(self):
        assert self.ended

        return self.cumulative_reward

    def step(self, action):
        assert not self.ended and self.is_legal_action(action)

        self.state, reward, self.ended, _ = self.env.step(action.index)
        self.cumulative_reward += reward
        return reward

    def get_state(self):
        return self.state


# class CartPoleGame(Game):
#     def __init__(self, **game_params):
#         super().__init__(environment=CartPoleEnvironment(**game_params))
#         self.history.observations.append(self.make_image())
#
#     def state_repr(self, state_index=-1):
#         # self.environment.env.render()
#         return 'Cart position: {}\nCart velocity: {}\nPole angle: {}\nPole velocity at tip: {}'.format(*self.history.observations[state_index])
#
#     def make_image(self):
#         return np.append(self.environment.get_state(), len(self.history)/self.environment.max_moves).astype(np.float32)


class CartPoleNetwork(Network):
    """
    Neural networks for cart-pole game.
    """

    def __init__(self, state_action_encoder,
                 num_layers, num_units,  # Fully connected network parameters
                 policy_filters, policy_kernel_size,  # Policy head parameters
                 value_filters, value_kernel_size,  # Value head parameters
                 reward_filters, reward_kernel_size,  # Reward head parameters
                 hidden_size, scalar_activation  # For value and reward heads
                 ):
        """
        Representation input (observation batch):       (batch_size, 5).
        Representation output (hidden state batch):     (batch_size, num_units, 1, 1)

        Encoded action batch:                           (batch_size, num_units, 1, 1)

        Dynamics input:                                 (batch_size, num_units, 1, 2)
        Dynamics outputs:
            - batch_hidden_state:                       (batch_size, num_units, 1, 1)
            - batch_reward:                             (batch_size, num_players=1)
            - batch_toplay:                             (batch_size, num_players=1)

        Prediction input:                               (batch_size, num_units, 1, 1)
        Prediction outputs:
            - batch_policy_logits:                      (batch_size, action_space_size=2)
            - batch_value:                              (batch_size, num_players=1)
        """

        super().__init__(state_action_encoder=state_action_encoder)

        self.representation = fully_connected_representation_network(name='CPRep', input_shape=(4,),
                                                                     num_layers=num_layers, num_units=num_units)

        self.dynamics = fully_connected_dynamics_network(name='CPDyn', input_shape=(num_units, 1, 2),
                                                         num_layers=num_layers, num_units=num_units,
                                                         reward_filters=reward_filters, reward_kernel_size=reward_kernel_size,
                                                         reward_hidden_size=hidden_size, scalar_activation=scalar_activation)

        self.prediction = fully_connected_prediction_network(name='CPPre', input_shape=(num_units, 1, 1), num_logits=2,
                                                             num_layers=num_layers, num_units=num_units,
                                                             policy_filters=policy_filters, policy_kernel_size=policy_kernel_size,
                                                             value_filters=value_filters, value_kernel_size=value_kernel_size,
                                                             value_hidden_size=hidden_size, scalar_activation=scalar_activation)

        self.state_action_encoding = state_action_encoder

        self.trainable_variables = []
        for sub_network in [self.representation, self.dynamics, self.prediction]:
            self.trainable_variables.extend(sub_network.trainable_variables)

    def hidden_state_shape(self, batch_size=None):
        input_shape = list(self.prediction.input_shape)
        input_shape[0] = batch_size
        return tuple(input_shape)
