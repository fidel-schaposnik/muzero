from math import log

from dataclasses import dataclass
from muzero.utils import hparams_safe_update, scalar_to_support, support_to_scalar

# For type annotations
from typing import Dict, Callable, Any, Optional
import tensorflow as tf

from muzero.muprover_types import Action, Value
from muzero.environment import Environment
from muzero.utils import KnownBounds


class ScalarConfig:
    def __init__(self,
                 known_bounds: Optional[KnownBounds],
                 support_size: Optional[int],
                 loss_decay: float
                 ) -> None:
        self.known_bounds: Optional[KnownBounds] = known_bounds
        self.support_size: Optional[int] = support_size
        self.loss_decay: float = loss_decay

        if support_size is None:
            self.loss: tf.keras.losses.Loss = tf.keras.losses.MeanSquaredError()
            self.inv_to_scalar: Callable[[tf.Tensor], tf.Tensor] = lambda tensor: tf.expand_dims(tensor, axis=-1)
            self.to_scalar: Callable[[tf.Tensor], tf.Tensor] = lambda tensor: tensor
        else:
            self.loss: tf.keras.losses.Loss = tf.keras.losses.CategoricalCrossentropy()
            self.inv_to_scalar: Callable[[tf.Tensor], tf.Tensor] = self.scalar_to_support
            self.to_scalar: Callable[[tf.Tensor], tf.Tensor] = self.support_to_scalar

    def scalar_to_support(self, tensor: tf.Tensor) -> tf.Tensor:
        a, b = self.known_bounds if self.known_bounds is not None else (0.0, 1.0)
        return scalar_to_support((tensor-a)/(b-a), support_size=self.support_size)

    def support_to_scalar(self, tensor: tf.Tensor) -> tf.Tensor:
        a, b = self.known_bounds if self.known_bounds is not None else (0.0, 1.0)
        return a + (b-a)*support_to_scalar(tensor, support_size=self.support_size)


@dataclass
class GameConfig:
    """
    game definition
    """
    name: str
    environment_class: Callable[..., Environment]
    environment_parameters: Dict[str, Any]
    action_space_size: int
    discount: float


@dataclass
class ReplayBufferConfig:
    """replay buffer config

    window_size: the number of most recent games from which replays are sampled
    
    prefetch_buffer_size: controls the number of batches the replay
            buffer prepares in advance, to be ready before they are
            required by the training process. This enters as a
            parameter directly in a tf.data.Dataset (
            https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch
            ), and should improve performance when actual preparation
            of a batch takes a long time (e.g. because batch_size is
            large), effectively making it run asynchronously with
            training.
    """
    window_size: int
    prefetch_buffer_size: int


@dataclass
class TrainingConfig:
    """
    Training configuration.
    """
    optimizer: tf.keras.optimizers.Optimizer
    batch_size: int
    training_steps: int
    checkpoint_interval: int
    replay_buffer_loginterval: int
    num_unroll_steps: int
    td_steps: int
    steps_per_execution: int


@dataclass
class MCTSConfig:
    """
    MCTS configuration
    """
    max_moves: int
    root_dirichlet_alpha: float
    root_exploration_fraction: float
    num_simulations: int
    temperature: float
    freezing_moves: int
    default_value: Value


class NetworkConfig:
    REPRESENTATION = 'representation'
    DYNAMICS = 'dynamics'
    PREDICTION = 'prediction'
    INITIAL_INFERENCE = 'initial_inference'
    RECURRENT_INFERENCE = 'recurrent_inference'
    UNROLLED_MODEL = 'unrolled_model'
    OBSERVATION = 'observation'
    ACTION = 'action'
    HIDDEN_STATE = 'hidden_state'
    UNROLL_ACTIONS = 'unroll_actions'
    UNROLLED_REWARDS = 'unrolled_rewards'
    UNROLLED_VALUES = 'unrolled_values'
    UNROLLED_POLICY_LOGITS = 'unrolled_policy_logits'

    def __init__(self,
                 network_class: Callable,
                 **network_parameters
                 ) -> None:
        self.network_class: Callable = network_class
        self.network_parameters: Dict[str, Any] = network_parameters


class MuZeroConfig:
    def __init__(self, game_config: GameConfig,
                 replay_buffer_config: ReplayBufferConfig,
                 training_config: TrainingConfig,
                 mcts_config: MCTSConfig,
                 network_config: NetworkConfig,
                 reward_config: ScalarConfig,
                 value_config: ScalarConfig) -> None:
        self.game_config: GameConfig = game_config
        self.replay_buffer_config: ReplayBufferConfig = replay_buffer_config
        self.training_config: TrainingConfig = training_config
        self.mcts_config: MCTSConfig = mcts_config
        self.network_config: NetworkConfig = network_config
        self.reward_config: ScalarConfig = reward_config
        self.value_config: ScalarConfig = value_config

    def action_space(self):
        return [Action(index) for index in range(self.game_config.action_space_size)]

    def create_environment(self) -> Environment:
        return self.game_config.environment_class(**self.game_config.environment_parameters)

    def make_uniform_network(self):  # TODO: How do we avoid circular dependencies with type-hinting?
        return self.network_config.network_class(self, **self.network_config.network_parameters)

    def visit_softmax_temperature_fn(self, num_moves: int, training_steps: int) -> float:
        if num_moves < self.mcts_config.freezing_moves:
            return self.mcts_config.temperature
        else:
            return 0.0

    def exploration_function(self, parent_visit_count: int, child_visit_count: int) -> float:
        return 2 * log(parent_visit_count + 1) / (child_visit_count + 1)

    def hyperparameters(self):
        hyperparameters = {}
        hparams_safe_update(hyperparameters, self.game_config.environment_parameters)
        hparams_safe_update(hyperparameters, self.network_config.network_parameters)
        hparams_safe_update(hyperparameters, self.training_config.optimizer.get_config())

        hparams_safe_update(hyperparameters, vars(self.game_config))
        hparams_safe_update(hyperparameters, vars(self.replay_buffer_config))
        hparams_safe_update(hyperparameters, vars(self.network_config))
        hparams_safe_update(hyperparameters, vars(self.training_config))
        hparams_safe_update(hyperparameters, vars(self.mcts_config))
        return hyperparameters
