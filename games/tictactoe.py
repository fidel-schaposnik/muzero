from game import *
from network import *
from config import *
import tensorflow as tf


def make_config(window_size=int(1e3), batch_size=2048,
                training_steps=int(1e4), checkpoint_interval=int(1e2), num_simulations=27,
                conv_filters=32, conv_kernel_size=(3, 3), tower_height=4,  # Residual tower parameters
                policy_filters=64, policy_kernel_size=(3, 3),  # Policy sub-network parameters
                value_filters=64, value_kernel_size=(3, 3),  # Value sub-network parameters
                reward_filters=64, reward_kernel_size=(3, 3),  # Reward sub-network parameters
                toplay_filters=64, toplay_kernel_size=(3, 3),  # To-play sub-network parameters
                hidden_size=128  # Parameters shared by value and reward sub-networks
                ):

    return MuZeroConfig(name='TicTacToe',
                        value_loss_decay=1.0,
                        reward_loss_decay=1.0,
                        regularization_decay=1e-4,
                        window_size=window_size,
                        batch_size=batch_size,
                        num_unroll_steps=5,
                        td_steps=9,
                        training_steps=training_steps,
                        checkpoint_interval=checkpoint_interval,
                        learning_rate=.001,
                        num_simulations=num_simulations,
                        known_bounds=None,
                        discount=0.9,
                        freezing_moves=10,
                        root_dirichlet_alpha=0.25,
                        root_exploration_noise=0.1,
                        max_moves=9,
                        game_class=TicTacToeGame,
                        network_class=TicTacToeNetwork,
                        action_space_size=9,
                        conv_filters=conv_filters, conv_kernel_size=conv_kernel_size, tower_height=tower_height,
                        policy_filters=policy_filters, policy_kernel_size=policy_kernel_size,
                        value_filters=value_filters, value_kernel_size=value_kernel_size,
                        reward_filters=reward_filters, reward_kernel_size=reward_kernel_size,
                        toplay_filters=toplay_filters, toplay_kernel_size=toplay_kernel_size,
                        hidden_size=hidden_size
                        )


class TicTacToeEnvironment(Environment):
    """
    The environment class of tic-tac-toe.
    """

    def __init__(self, **kwargs):  # kwargs collects arguments not used here (network parameters)
        """
        Create the environment where tic-tac-toe is played, initialize to an empty board.
        """
        super().__init__(action_space_size=9, num_players=2)

        # Game state
        self.board = -np.ones(shape=(3, 3), dtype=np.int)  # -1 if cell is empty, player_id if cell is filled
        self.steps = 0
        self.ended = False
        self.winner = None

        # Rewards
        self.WIN = 1.0
        self.LOSE = -1.0
        self.DRAW = 0.0

    def is_legal_action(self, action):
        row = action.index // 3
        col = action.index % 3
        return self.board[row, col] == -1

    def to_play(self):
        return Player(self.steps % 2)

    def terminal(self):
        return self.ended

    def outcome(self):
        assert self.ended

        if self.winner:
            return {player: self.WIN if player == self.winner else self.LOSE for player in self.players()}
        else:
            return {player: self.DRAW for player in self.players()}

    def step(self, action):
        assert not self.ended and self.is_legal_action(action)

        # Find the position of this move
        row = action.index // 3
        col = action.index % 3
        player_id = self.to_play().player_id

        # Make the move
        self.board[row, col] = player_id
        self.steps += 1

        # Check if the game ended
        for i in range(3):
            if np.all(self.board[i, :] == player_id) or np.all(self.board[:, i] == player_id):
                self.ended = True
        if np.all(self.board.diagonal() == player_id) or np.all(np.flip(self.board, axis=0).diagonal() == player_id):
            self.ended = True
        if self.ended:
            self.winner = Player(player_id)

        if self.steps == 9:
            # Game ends in draw
            self.ended = True
        if self.ended:
            return self.outcome()
        else:
            return {player: 0.0 for player in self.players()}

    def get_state(self):
        return [np.where(self.board == 0, 1, 0), np.where(self.board == 1, 1, 0)]


class TicTacToeGame(Game):
    def __init__(self, **game_params):
        super().__init__(environment=TicTacToeEnvironment(**game_params))
        self.history.observations.append(self.make_image())

    def state_repr(self, state_index=-1):
        board = self.history.observations[state_index][:,:,0]-self.history.observations[state_index][:,:,1]
        colors = {0: '.', 1: 'X', -1: 'O'}
        result = ''
        for i in range(3):
            for j in range(3):
                result += colors[board[i, j]]
            result += '\n'
        return result

    def make_image(self):
        return np.transpose(np.array(self.environment.get_state()), (1, 2, 0)).astype(np.float32)


class TicTacToeNetwork(Network):
    """
    Neural networks for tic-tac-toe game.
    """

    def __init__(self,
                 conv_filters, conv_kernel_size, tower_height,  # Residual tower parameters
                 policy_filters, policy_kernel_size,  # Policy head parameters
                 value_filters, value_kernel_size,  # Value head parameters
                 reward_filters, reward_kernel_size,  # Reward head parameters
                 toplay_filters, toplay_kernel_size,  # To-play head parameters
                 hidden_size,  # For value and reward heads
                 **kwargs  # Collects other parameters not used here (mostly for game definition)
                 ):
        """
        Initial inference:
            Observation batch:          (batch_size, 3, 3, num_players=2).
            Representation output:      (batch_size, 3, 3, conv_filters)
            Prediction outputs:
                - batch_policy_logits:  (batch_size, action_space_size=9)
                - batch_value:          (batch_size, num_players=2)

        Recurrent inference:
            Hidden state batch shape:   (batch_size, 3, 3, conv_filters)
            Encoded action batch shape: (batch_size, 3, 3, action_space_size=9)
            Dynamics input:             (batch_size, 3, 3, conv_filters+9)
            Dynamics outputs:
                - batch_hidden_state:   (batch_size, 3, 3, conv_filters)
                - batch_reward:         (batch_size, num_players=2)
                - batch_toplay:         (batch_size, num_players=2)
            Prediction input:           (batch_size, 3, 3, conv_filters)
            Prediction outputs:
                - batch_policy_logits:  (batch_size, action_space_size=9)
                - batch_value:          (batch_size, num_players=2)
        """

        super().__init__()
        self.encoded_action_space = np.zeros((9, 3, 3, 9)).astype(np.float32)
        for i in range(9):
            self.encoded_action_space[i, :, :, i] = 1

        self.representation = representation_network(name='TTTRep', input_shape=(3, 3, 2),
                                                     tower_height=tower_height, conv_filters=conv_filters, conv_kernel_size=conv_kernel_size)
        # self.representation = dummy_network(name='TTTRep', input_shape=(3, 3, 2), conv_filters=conv_filters)

        self.dynamics = dynamics_network(name='TTTDyn', input_shape=(3, 3, conv_filters + 9), num_players=2,
                                         tower_height=tower_height, conv_filters=conv_filters, conv_kernel_size=conv_kernel_size,
                                         reward_filters=reward_filters, reward_kernel_size=reward_kernel_size, reward_hidden_size=hidden_size,
                                         toplay_filters=toplay_filters, toplay_kernel_size=toplay_kernel_size)

        self.prediction = prediction_network(name='TTTPre', input_shape=(3, 3, conv_filters), num_logits=9, num_players=2,
                                             tower_height=tower_height, conv_filters=conv_filters, conv_kernel_size=conv_kernel_size,
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
        batch_encoded_action = self.encoded_action_space[[action.index for action in batch_action]]

        # Concatenate action to hidden state
        batch_dynamics_input = tf.concat([batch_hidden_state, batch_encoded_action], axis=-1)
        return batch_dynamics_input
