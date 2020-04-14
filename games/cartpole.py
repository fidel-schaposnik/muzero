from game import *
from network import *
from config import *
import tensorflow as tf
import gym


def make_config(window_size=int(1e3), batch_size=2048,
                training_steps=int(1e4), checkpoint_interval=int(1e2), num_simulations=10,
                num_layers=2, num_units=128,  # Fully connected sub-network parameters
                policy_filters=32, policy_kernel_size=(3, 3),  # Policy sub-network parameters
                value_filters=32, value_kernel_size=(3, 3),  # Value sub-network parameters
                reward_filters=32, reward_kernel_size=(10, 1),  # Reward sub-network parameters
                toplay_filters=32, toplay_kernel_size=(10, 1),  # To-play sub-network parameters
                hidden_size=32  # Parameters shared by value and reward sub-networks
                ):

    return MuZeroConfig(name='CartPole',
                        value_loss_decay=1.0,
                        reward_loss_decay=1.0,
                        regularization_decay=1e-4,
                        window_size=window_size,
                        batch_size=batch_size,
                        num_unroll_steps=5,
                        td_steps=10,
                        training_steps=training_steps,
                        checkpoint_interval=checkpoint_interval,
                        learning_rate=.001,
                        num_simulations=num_simulations,
                        known_bounds=None,
                        discount=0.9,
                        freezing_moves=50,
                        root_dirichlet_alpha=0.25,
                        root_exploration_noise=0.1,
                        max_moves=200,
                        game_class=CartPoleGame,
                        network_class=CartPoleNetwork,
                        action_space_size=2,
                        num_layers=num_layers, num_units=num_units,
                        policy_filters=policy_filters, policy_kernel_size=policy_kernel_size,
                        value_filters=value_filters, value_kernel_size=value_kernel_size,
                        reward_filters=reward_filters, reward_kernel_size=reward_kernel_size,
                        toplay_filters=toplay_filters, toplay_kernel_size=toplay_kernel_size,
                        hidden_size=hidden_size
                        )


class CartPoleEnvironment(Environment):
    """
    The environment class of cart-pole.
    """

    def __init__(self, **kwargs):  # kwargs collects arguments not used here (network parameters)
        super().__init__(action_space_size=2, num_players=1)
        self.env = gym.make('CartPole-v0')
        self.state = self.env.reset()
        self.ended = False
        self.cumulative_reward = 0.0

    def is_legal_action(self, action):
        return True

    def to_play(self):
        return Player(0)

    def terminal(self):
        return self.ended

    def outcome(self):
        assert self.ended

        return {Player(0): self.cumulative_reward}

    def step(self, action):
        assert not self.ended and self.is_legal_action(action)

        self.state, reward, self.ended, _ = self.env.step(action.index)
        self.cumulative_reward += reward
        return {Player(0): reward}

    def get_state(self):
        return self.state


class CartPoleGame(Game):
    def __init__(self, **game_params):
        super().__init__(environment=CartPoleEnvironment(**game_params))
        self.history.observations.append(self.make_image())

    def state_repr(self, state_index=-1):
        return 'Cart position: {}\nCart velocity: {}\nPole angle: {}\nPole velocity at tip: {}'.format(*self.history.observations[state_index])

    def make_image(self):
        return self.environment.get_state().astype(np.float32)


class CartPoleNetwork(Network):
    """
    Neural networks for cart-pole game.
    """

    def __init__(self,
                 num_layers, num_units,  # Fully connected network parameters
                 policy_filters, policy_kernel_size,  # Policy head parameters
                 value_filters, value_kernel_size,  # Value head parameters
                 reward_filters, reward_kernel_size,  # Reward head parameters
                 toplay_filters, toplay_kernel_size,  # To-play head parameters
                 hidden_size,  # For value and reward heads
                 **kwargs  # Collects other parameters not used here (mostly for game definition)
                 ):
        """
        Representation input (observation batch):       (batch_size, 4).
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

        super().__init__()

        self.representation = fully_connected_representation_network(name='CPRep', input_shape=(4,),
                                                                     num_layers=num_layers, num_units=num_units)

        self.dynamics = fully_connected_dynamics_network(name='CPDyn', input_shape=(num_units,1,2), num_players=1,
                                                         num_layers=num_layers, num_units=num_units,
                                                         reward_filters=reward_filters, reward_kernel_size=reward_kernel_size, reward_hidden_size=hidden_size,
                                                         toplay_filters=toplay_filters, toplay_kernel_size=toplay_kernel_size)

        self.prediction = fully_connected_prediction_network(name='CPPre', input_shape=(num_units,1,1), num_logits=2, num_players=1,
                                                             num_layers=num_layers, num_units=num_units,
                                                             policy_filters=policy_filters, policy_kernel_size=policy_kernel_size,
                                                             value_filters=value_filters, value_kernel_size=value_kernel_size, value_hidden_size=hidden_size)

        self.trainable_variables = []
        for sub_network in [self.representation, self.dynamics, self.prediction]:
            self.trainable_variables.extend(sub_network.trainable_variables)

    def hidden_state_shape(self, batch_size=None):
        input_shape = list(self.prediction.input_shape)
        input_shape[0] = batch_size
        return tuple(input_shape)

    def toplay_shape(self, batch_size=None):
        output_shape = list(self.dynamics.output_shape[-1])
        output_shape[0] = batch_size
        return tuple(output_shape)

    def state_action_encoding(self, batch_hidden_state, batch_action):
        # Encode action as binary planes
        batch_encoded_action = np.zeros_like(batch_hidden_state)
        for i, action in enumerate(batch_action):
            batch_encoded_action[i] = action.index

        # Concatenate action to hidden state
        batch_dynamics_input = tf.concat([batch_hidden_state, batch_encoded_action], axis=-1)
        return batch_dynamics_input
