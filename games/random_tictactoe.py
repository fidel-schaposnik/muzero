import random
import numpy as np
import tensorflow as tf

from muzero.config import MuZeroConfig, GameConfig, ReplayBufferConfig, MCTSConfig, NetworkConfig, TrainingConfig, ScalarConfig
from muzero.utils import KnownBounds
from muzero.environment import Environment
from muzero.exceptions import MuProverEnvironmentError
from muzero.network import Network, one_hot_tensor_encoder, dummy_model

# For type annotations
from typing import Optional, Tuple

from muzero.muprover_types import State, Value, Action


def make_config() -> MuZeroConfig:
    game_config = GameConfig(name='RandomTicTacToe',
                             environment_class=RandomTicTacToeEnvironment,
                             environment_parameters={},
                             action_space_size=9,
                             discount=1.0
                             )

    replay_buffer_config = ReplayBufferConfig(window_size=int(1e4),
                                              prefetch_buffer_size=10
                                              )

    mcts_config = MCTSConfig(max_moves=9,
                             root_dirichlet_alpha=1.0,
                             root_exploration_fraction=0.25,
                             num_simulations=20,
                             temperature=1.0,
                             freezing_moves=9,
                             default_value=Value(0.5)
                             )

    network_config = NetworkConfig(network_class=TicTacToeNetwork,
                                   regularizer=tf.keras.regularizers.l2(l=1e-4),
                                   hidden_state_size=128,
                                   hidden_size=128
                                   )

    training_config = TrainingConfig(optimizer=tf.keras.optimizers.Adam(),
                                     batch_size=128,
                                     training_steps=int(5e4),
                                     checkpoint_interval=int(5e2),
                                     replay_buffer_loginterval=50,
                                     num_unroll_steps=2,
                                     td_steps=9,
                                     steps_per_execution=1
                                     )

    reward_config = ScalarConfig(known_bounds=KnownBounds(min=-1.0, max=1.0),
                                 support_size=None,
                                 loss_decay=0.0
                                 )

    value_config = ScalarConfig(known_bounds=KnownBounds(min=-1.0, max=1.0),
                                support_size=None,
                                loss_decay=4.0)

    return MuZeroConfig(game_config=game_config,
                        replay_buffer_config=replay_buffer_config,
                        mcts_config=mcts_config,
                        training_config=training_config,
                        network_config=network_config,
                        value_config=value_config,
                        reward_config=reward_config)


class RandomTicTacToeEnvironment(Environment):
    """
    The environment class of tic-tac-toe.
    """

    def __init__(self, player_choice: Optional[tf.Tensor] = None) -> None:
        super().__init__(action_space_size=9)
        self.player_choice = player_choice

        # Game state
        self.board: Optional[np.ndarray] = None  # numpy array of shape (3,3), -1 if empty or player_id if filled
        self.steps: Optional[int] = None         # number of steps taken in this episode
        self.ended: Optional[bool] = None        # flag to mark the episode has finished
        self.winner: Optional[int] = None        # contains the player_id of the winner if the episode has finished
        self.player_id: Optional[int] = None     # player_id of the player (the opponent is a random player)

        # Game results
        self.WIN: Value = Value(1.0)
        self.DRAW: Value = Value(0.0)
        self.LOSE: Value = Value(-1.0)

    def __repr__(self) -> str:
        colors = {-1: '.', 0: 'X', 1: 'O'}
        result = ''
        if self.board is not None:
            for i in range(3):
                for j in range(3):
                    result += colors[self.board[i, j]]
                result += '\n'
        return result

    def _get_state(self) -> State:
        return tf.stack([tf.where(self.board == 0, 1., 0.),
                         tf.where(self.board == 1, 1., 0.),
                         self.player_id * tf.ones(shape=(3, 3))], axis=-1)

    def _check_last_move(self, player_id: int) -> bool:
        for i in range(3):
            if np.all(self.board[i, :] == player_id) or np.all(self.board[:, i] == player_id):
                self.ended = True
        if np.all(self.board.diagonal() == player_id) or np.all(np.flip(self.board, axis=0).diagonal() == player_id):
            self.ended = True
        if self.ended:
            self.winner = player_id
        elif self.steps == 9:
            self.ended = True
        return self.ended

    def _make_random_move(self, player_id: int) -> None:
        valid_moves = [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == -1]
        row, col = random.choice(valid_moves)
        self.board[row, col] = player_id
        self.steps += 1
        self._check_last_move(player_id)

    def step(self, action: Action) -> Tuple[State, Value, bool, dict]:
        if not self.is_legal_action(action):
            raise MuProverEnvironmentError(f'cannot perform action {action}')

        # Find the position of this move
        row = action // 3
        col = action % 3

        # Make the move
        self.board[row, col] = self.player_id
        self.steps += 1

        self._check_last_move(self.player_id)
        if not self.ended:
            self._make_random_move(1-self.player_id)

        reward = self.DRAW if self.winner is None else (self.WIN if self.winner == self.player_id else self.LOSE)
        return self._get_state(), reward, self.ended, {}

    def reset(self) -> State:
        self.board = -np.ones(shape=(3, 3), dtype=np.int)  # -1 is empty, player_id if filled
        self.steps = 0
        self.ended = False
        self.winner = None
        self.player_id = self.player_choice.numpy() if self.player_choice is not None else random.randrange(2)
        if self.player_id == 1:
            self._make_random_move(0)
        return self._get_state()

    def is_legal_action(self, action: Action) -> bool:
        if self.ended or action not in range(9):
            return False
        row = action // 3
        col = action % 3
        return self.board[row, col] == -1


class TicTacToeNetwork(Network):
    """
    Neural networks for tic-tac-toe game.
    """

    def __init__(self,
                 config: MuZeroConfig,
                 regularizer: tf.keras.regularizers.Regularizer,
                 hidden_state_size: int,
                 hidden_size: int
                 ) -> None:
        """
        Representation input (observation batch):       (batch_size, 3, 3, 3).
        Representation output (hidden state batch):     (batch_size, 3, 3, hidden_state_size)

        Encoded action batch:                           (batch_size, 3, 3, hidden_state_size+9)

        Dynamics input:                                 (batch_size, 3, 3, hidden_state_size+9)
        Dynamics outputs:
            - hidden_state:                             (batch_size, 3, 3, hidden_state_size)
            - reward:                                   (batch_size, )

        Prediction input:                               (batch_size, 3, 3, hidden_state_size)
        Prediction outputs:
            - policy_logits:                            (batch_size, action_space_size=9)
            - value:                                    (batch_size, )
        """

        random_tictactoe_state_preprocessing: tf.keras.Model = dummy_model(input_shape=(3, 3, 3))

        random_tictactoe_representation: tf.keras.Model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=3, padding='same', activation='relu',
                                   input_shape=(3, 3, 3), kernel_regularizer=regularizer, bias_regularizer=regularizer),
            tf.keras.layers.Dense(units=hidden_state_size, activation='relu',
                                  kernel_regularizer=regularizer, bias_regularizer=regularizer)
        ], name=config.network_config.REPRESENTATION)

        encoded_state_action = tf.keras.Input(shape=(3, 3, hidden_state_size + 9))
        x = tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=3, padding='same', activation='relu',
                                   kernel_regularizer=regularizer, bias_regularizer=regularizer)(encoded_state_action)
        hidden_state = tf.keras.layers.Dense(units=hidden_state_size, activation='relu',
                                             kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        x = tf.keras.layers.Flatten()(hidden_state)
        reward_output = tf.keras.layers.Dense(units=1, activation='tanh', name='reward',
                                              kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        random_tictactoe_dynamics: tf.keras.Model = tf.keras.Model(inputs=encoded_state_action,
                                                                   outputs=[hidden_state, reward_output],
                                                                   name=config.network_config.DYNAMICS)

        hidden_state = tf.keras.Input(shape=(3, 3, hidden_state_size))
        x = tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=3, activation='relu',
                                   kernel_regularizer=regularizer, bias_regularizer=regularizer)(hidden_state)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=hidden_size, activation='relu',
                                  kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        value_output = tf.keras.layers.Dense(units=1, activation='tanh', name='value',
                                             kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        policy_logits_output = tf.keras.layers.Dense(units=9, name='policy_logits',
                                                     kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)

        random_tictactoe_prediction: tf.keras.Model = tf.keras.Model(inputs=hidden_state,
                                                                     outputs=[value_output, policy_logits_output],
                                                                     name=config.network_config.PREDICTION)

        super().__init__(config=config,
                         representation=random_tictactoe_representation,
                         dynamics=random_tictactoe_dynamics,
                         prediction=random_tictactoe_prediction,
                         state_action_encoder=one_hot_tensor_encoder(state_shape=(3, 3), action_space_size=9),
                         state_preprocessing=random_tictactoe_state_preprocessing)
