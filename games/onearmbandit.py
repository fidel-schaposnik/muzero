from config import *
from game import *
from network import *
from utils import *


def make_config(x=None, z=0.5, t=20, scalar_support_size=100,
                window_size=int(1e3), batch_size=2048, td_steps=100, discount=0.9, max_moves=100,
                training_steps=int(1e4), checkpoint_interval=int(1e3), num_simulations=16,
                conv_filters=32, conv_kernel_size=(3, 3), tower_height=4,  # Residual tower parameters
                policy_filters=32, policy_kernel_size=(3, 3),  # Policy sub-network parameters
                value_filters=32, value_kernel_size=(3, 3),  # Value sub-network parameters
                reward_filters=32, reward_kernel_size=(3, 3),  # Reward sub-network parameters
                toplay_filters=1, toplay_kernel_size=(1, 1),  # To-play sub-network parameters
                hidden_size=32  # Parameters shared by value and reward sub-networks
                ):

    return MuZeroConfig(name='OneArmBandit',
                        weight_decay=1e-4,
                        window_size=window_size,
                        batch_size=batch_size,
                        num_unroll_steps=5,
                        td_steps=td_steps,
                        training_steps=training_steps,
                        checkpoint_interval=checkpoint_interval,
                        learning_rate=.001,
                        num_simulations=num_simulations,
                        known_bounds=None,
                        discount=discount,
                        freezing_moves=100,
                        root_dirichlet_alpha=0.25,
                        root_exploration_noise=0.1,
                        max_moves=max_moves,
                        game_class=OneArmBanditGame,
                        network_class=OneArmBanditNetwork,
                        action_space_size=2,
                        scalar_support_size=scalar_support_size,
                        conv_filters=conv_filters, conv_kernel_size=conv_kernel_size, tower_height=tower_height,
                        policy_filters=policy_filters, policy_kernel_size=policy_kernel_size,
                        value_filters=value_filters, value_kernel_size=value_kernel_size,
                        reward_filters=reward_filters, reward_kernel_size=reward_kernel_size,
                        toplay_filters=toplay_filters, toplay_kernel_size=toplay_kernel_size,
                        hidden_size=hidden_size,
                        x=x,
                        z=z,
                        t=t
                        )


class OneArmBanditEnvironment(Environment):
    """
    The environment of the one-arm bandit game.
    """

    def __init__(self, x, z, t, **kwargs):  # **kwargs collects arguments not used here, mostly network parameters
        """
        Create the environment where one-arm bandit is played.
        """
        super().__init__(action_space_size=2, num_players=1)

        # Game parameters
        self.X = random.random() if not x else x
        self.Z = z
        self.T = t

        # Game state
        self.steps = 0
        self.ended = False
        self.rewards = []

    def to_play(self) -> Player:
        return Player(0)

    def is_legal_action(self, action: Action) -> bool:
        return True

    def terminal(self):
        return self.ended

    def outcome(self):
        return sum(self.rewards)

    def step(self, action):
        assert not self.ended and self.is_legal_action(action)

        # Make the move
        self.steps += 1
        if self.steps == self.T:
            self.ended = True

        if action.index == 0:
            reward = self.Z
        else:
            reward = 1.0 if random.random() < self.X else 0.0
        self.rewards.append(reward)

        return {Player(0): reward}

    def get_state(self):
        return self.rewards

    def print_state(self):
        print('Total reward handed out so far: {:.2f}'.format(self.outcome()))


class OneArmBanditGame(Game):
    def __init__(self, scalar_support_size, **game_params):
        super().__init__(environment=OneArmBanditEnvironment(**game_params))

        self.scalar_support_size = scalar_support_size
        self.history.observations.append(self.make_image())

    def make_image(self):
        state = self.environment.get_state()
        padded_state = np.pad(state, (self.environment.T-len(state), 0))
        return np.expand_dims(scalar_to_support(padded_state, self.scalar_support_size), -1)


class OneArmBanditNetwork(Network):
    """
    Neural networks for one-arm bandit game.
    """

    def __init__(self, t, scalar_support_size,
                 conv_filters, conv_kernel_size, tower_height,  # Residual tower parameters
                 policy_filters, policy_kernel_size,  # Policy head parameters
                 value_filters, value_kernel_size,  # Value head parameters
                 reward_filters, reward_kernel_size,  # Reward head parameters
                 toplay_filters, toplay_kernel_size,  # To-play head parameters
                 hidden_size,  # For value and reward heads
                 **kwargs  # Collects other parameters not used here (mostly for game definition)
                 ):
        """
        Create residual networks for representation, dynamics and prediction functions
        """

        super().__init__()
        self.encoded_action_space = np.zeros((2, t, scalar_support_size+1, 2)).astype(np.float32)
        for i in range(2):
            self.encoded_action_space[i, :, :, i] = 1

        self.inv_scalar_transformation = lambda x_categorical: support_to_scalar(x_categorical, scalar_support_size)

        self.representation = representation_network(name='OABRep', input_shape=(t, scalar_support_size+1, 1),
                                                     tower_height=tower_height, conv_filters=conv_filters, conv_kernel_size=conv_kernel_size)

        self.dynamics = categorical_dynamics_network(
            name='OABDyn', input_shape=(t, scalar_support_size+1, conv_filters+2), num_players=1,
            tower_height=tower_height, conv_filters=conv_filters, conv_kernel_size=conv_kernel_size,
            scalar_support_size=scalar_support_size, reward_filters=reward_filters, reward_kernel_size=reward_kernel_size, reward_hidden_size=hidden_size,
            toplay_filters=toplay_filters, toplay_kernel_size=toplay_kernel_size)

        self.prediction = categorical_prediction_network(
            name='OABPre', input_shape=(t, scalar_support_size+1, conv_filters),
            tower_height=tower_height, conv_filters=conv_filters, conv_kernel_size=conv_kernel_size,
            num_logits=2, policy_filters=policy_filters, policy_kernel_size=policy_kernel_size,
            num_players=1, scalar_support_size=scalar_support_size, value_filters=value_filters, value_kernel_size=value_kernel_size, value_hidden_size=hidden_size)

        self.trainable_variables = []
        for sub_network in [self.representation, self.dynamics, self.prediction]:
            self.trainable_variables.extend(sub_network.trainable_variables)

    def hidden_state_shape(self, batch_size=None):
        """
        Returns the shape of a batch of hidden states with the current network parameters.
        """
        input_shape = list(self.prediction.input_shape)
        input_shape[0] = batch_size
        return tuple(input_shape)

    def toplay_shape(self, batch_size):
        output_shape = list(self.dynamics.output_shape[-1])
        output_shape[0] = batch_size
        return tuple(output_shape)

    def initial_inference(self, batch_image):
        """
        Observation batch:      (batch_size, t, scalar_support_size+1, 1).
        Representation output:  (batch_size, t, scalar_support_size+1, conv_filters)
        Prediction outputs:
            - batch_policy_logits:  (batch_size, action_space_size=2)
            - batch_value:          (batch_size, num_players=1)
        """

        batch_hidden_state = self.representation(batch_image)

        batch_policy_logits, batch_value = self.prediction(batch_hidden_state)

        return NetworkOutput(value=batch_value,
                             reward=tf.zeros_like(batch_value),
                             policy_logits=batch_policy_logits,
                             hidden_state=batch_hidden_state,
                             to_play=tf.zeros(self.toplay_shape(len(batch_image))),
                             inv_scalar_transformation=self.inv_scalar_transformation
                             )

    def recurrent_inference(self, batch_hidden_state, batch_action):
        """
        Hidden state batch shape:   (batch_size, t, scalar_support_size+1, conv_filters)
        Encoded action batch shape: (batch_size, t, scalar_support_size+1, action_space_size=2)
        Dynamics input:             (batch_size, t, scalar_support_size+1, conv_filters+2)
        Dynamics outputs:
            - batch_hidden_state:   (batch_size, t, scalar_support_size+1, conv_filters)
            - batch_reward:         (batch_size, num_players=1)
            - batch_toplay:         (batch_size, num_players=1)
        Prediction input:           (batch_size, t, scalar_support_size+1, conv_filters)
        Prediction outputs:
            - batch_policy_logits:  (batch_size, action_space_size=2)
            - batch_value:          (batch_size, num_players=1)
        """

        # Encode action as binary planes
        batch_encoded_action = self.encoded_action_space[[action.index for action in batch_action]]

        # Concatenate action to hidden state
        batch_dynamics_input = np.concatenate([batch_hidden_state, batch_encoded_action], axis=-1)

        # Apply the dynamics + prediction networks
        batch_hidden_state, batch_reward, batch_toplay = self.dynamics(batch_dynamics_input)

        batch_policy_logits, batch_value = self.prediction(batch_hidden_state)

        return NetworkOutput(value=batch_value,
                             reward=batch_reward,
                             policy_logits=batch_policy_logits,
                             hidden_state=batch_hidden_state,
                             to_play=batch_toplay,
                             inv_scalar_transformation=self.inv_scalar_transformation)
