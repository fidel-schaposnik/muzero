from game import *
from network import *
from config import *
import tensorflow as tf


def make_tictactoe_config(window_size=int(1e3), batch_size=2048,
                          training_steps=int(1e4), checkpoint_interval=int(1e3), num_simulations=25,
                          conv_filters=32, conv_kernel_size=(3, 3), tower_height=4,  # Residual tower parameters
                          policy_filters=64, policy_kernel_size=(3, 3),  # Policy sub-network parameters
                          value_filters=64, value_kernel_size=(3, 3),  # Value sub-network parameters
                          reward_filters=64, reward_kernel_size=(3, 3),  # Reward sub-network parameters
                          toplay_filters=64, toplay_kernel_size=(3, 3),  # To-play sub-network parameters
                          hidden_size=128  # Parameters shared by value and reward sub-networks
                          ):
    return MuZeroConfig(name='TicTacToe',
                        reward_loss_func=tf.keras.losses.mean_squared_error,
                        value_loss_func=tf.keras.losses.mean_squared_error,
                        weight_decay=1e-4,
                        window_size=window_size,
                        batch_size=batch_size,
                        num_unroll_steps=5,
                        td_steps=9,
                        training_steps=training_steps,
                        checkpoint_interval=checkpoint_interval,
                        learning_rate=.001,
                        num_simulations=num_simulations,
                        discount=1,
                        freezing_moves=6,
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
    """The environment of tic-tac-toe."""

    def __init__(self, **kwargs):  # kwargs collects arguments not used here (network parameters)
        """
        Create the environment where Tic-Tac-Toe is played
        """

        # Game parameters
        self.action_space_size = 9
        self.num_players = 2

        # Game state
        self.board = -np.ones(shape=(3, 3), dtype=np.int)
        self.steps = 0
        self.ended = False
        self.winner = None

    def is_legal_action(self, action: Action):
        row = action.index // 3
        col = action.index % 3
        return self.board[row, col] == -1

    def legal_actions(self):
        return [action for action in map(Action, range(self.action_space_size)) if self.is_legal_action(action)]

    def players(self):
        return [Player(i) for i in range(self.num_players)]

    def to_play(self):
        return Player(self.steps % 2)

    def terminal(self):
        return self.ended

    def outcome(self):
        return 1-2*self.winner.player_id if self.winner else 0

    def step(self, action: Action):
        """
        Make a move, return the reward.
        """
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
            if (self.board[i, :] == player_id).all() or (self.board[:, i] == player_id).all():
                self.ended = True
        if (self.board.diagonal() == player_id).all() or (np.flip(self.board, axis=0).diagonal() == player_id).all():
            self.ended = True
        if self.ended:
            self.winner = Player(player_id)
            return {player: (1.0 if player.player_id == player_id else 0.0) for player in self.players()}
            # return {player: (1.0 if player.player_id == player_id else -1.0) for player in self.players()}

        if self.steps == 9:
            # Game ends in draw
            self.ended = True
            return {player: 0.5 for player in self.players()}
        return {player: 0.0 for player in self.players()}

    def get_state(self):
        """
        Return the current state of the environment (in some canonical form, encoding is done elsewhere).
        """

        return [np.where(self.board == 0, 1, 0), np.where(self.board == 1, 1, 0)]

    def print_state(self):
        colors = {0: 'X', 1: 'O'}
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == -1:
                    print('.', end='')
                else:
                    print(colors[self.board[i, j]], end='')
            print()


class TicTacToeGame(Game):
    """For recording games of tic-tac-toe."""

    def __init__(self, **game_params):
        super().__init__()
        self.environment = TicTacToeEnvironment(**game_params)
        self.history = GameHistory(initial_state=self.make_image(), action_space_size=self.environment.action_space_size, num_players=self.environment.num_players)

    def make_image(self):
        return np.transpose(np.array(self.environment.get_state()), (1, 2, 0)).astype(np.float32)


class TicTacToeNetwork(Network):
    """
    Neural networks for tic-tac-toe game.
    """

    def __init__(self,
                 conv_filters=3, conv_kernel_size=(3, 3), tower_height=3,  # Residual tower parameters
                 policy_filters=2, policy_kernel_size=(1, 1),  # Policy sub-network parameters
                 value_filters=1, value_kernel_size=(1, 1),  # Value sub-network parameters
                 reward_filters=1, reward_kernel_size=(1, 1),  # Reward sub-network parameters
                 hidden_size=64,  # For value and reward sub-networks
                 **kwargs  # Collects other parameters not used here (mostly for game definition)
                 ):
        """
        Create residual networks for representation, dynamics and prediction functions
        """

        self.REPRESENTATION_TIME = 0.
        self.PREDICTION_TIME = 0.
        self.DYNAMICS_TIME = 0.
        self.RECINF_PREP_TIME = 0.

        super().__init__()
        self.encoded_action_space = np.zeros((9, 3, 3, 9)).astype(np.float32)
        for i in range(9):
            self.encoded_action_space[i, :, :, i] = 1

        self.representation = representation_network(name='TTTRep', input_shape=(3, 3, 2),
                                                     tower_height=tower_height, conv_filters=conv_filters, conv_kernel_size=conv_kernel_size)

        self.dynamics = dynamics_network(name='TTTDyn', input_shape=(3, 3, conv_filters + 9), num_players=2,
                                         tower_height=tower_height, conv_filters=conv_filters, conv_kernel_size=conv_kernel_size,
                                         reward_filters=reward_filters, reward_kernel_size=reward_kernel_size, reward_hidden_size=hidden_size)

        self.prediction = prediction_network(input_shape=(3, 3, conv_filters), name='TTTPre', num_logits=9, num_players=2,
                                             tower_height=tower_height, conv_filters=conv_filters, conv_kernel_size=conv_kernel_size,
                                             policy_filters=policy_filters, policy_kernel_size=policy_kernel_size,
                                             value_filters=value_filters, value_kernel_size=value_kernel_size, value_hidden_size=hidden_size)

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
        Takes a batch of images from Game.make_image, which has shape (batch_size, 3, 3, 2).
        Applies the representation network and performs a prediction.
        Returns a list of batch_size predictions packaged as a NetworkOutput object.
        """

        start = time.time()
        batch_hidden_state = self.representation(batch_image)
        end = time.time()
        self.REPRESENTATION_TIME += end - start

        start = time.time()
        batch_policy_logits, batch_value = self.prediction(batch_hidden_state)
        end = time.time()
        self.PREDICTION_TIME += end - start

        return NetworkOutput(batch_value, tf.zeros_like(batch_value), batch_policy_logits, batch_hidden_state, tf.zeros(self.toplay_shape(len(batch_image))))

    def recurrent_inference(self, batch_hidden_state, batch_action):
        """
        Takes a batch of hidden states (from representation or dynamics networks), which has shape (batch_size, 3, 3, conv_filters).
        Takes a batch of actions and encodes it as an array of shape (batch_size, 3, 3, action_space_size=9).
        Applies the dynamics network to the concatenation of both arrays above and performs a prediction.
        Returns a list of batch_size predictions of the form (value, reward, policy, hidden_state).
        """

        start = time.time()
        # Encode action as binary planes
        batch_encoded_action = self.encoded_action_space[[action.index for action in batch_action]]

        # Concatenate action to hidden state
        batch_dynamics_input = np.concatenate([batch_hidden_state, batch_encoded_action], axis=-1)
        end = time.time()
        self.RECINF_PREP_TIME += end - start

        # Apply the dynamics + prediction networks
        start = time.time()
        batch_hidden_state, batch_reward, batch_toplay = self.dynamics(batch_dynamics_input)
        end = time.time()
        self.DYNAMICS_TIME += end - start

        start = time.time()
        batch_policy_logits, batch_value = self.prediction(batch_hidden_state)
        end = time.time()
        self.PREDICTION_TIME += end - start

        return NetworkOutput(batch_value, batch_reward, batch_policy_logits, batch_hidden_state, batch_toplay)
