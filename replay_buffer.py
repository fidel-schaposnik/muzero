import random
import tensorflow as tf
from threading import RLock

from game_services import save_games, load_games
from utils import RollingMean

# For type annotations
from typing import Callable, List, Dict, Tuple, Optional, Any

from muzero_types import Observation, ActionBatch, ValueBatch, PolicyBatch, Value, Action
from config import MuZeroConfig
from game import GameHistory


class ReplayBuffer:
    """
    Buffer where games played by MuProver are stored for training purposes.
    """
    def __init__(self,
                 config: MuZeroConfig
                 ) -> None:
        self.config: MuZeroConfig = config
        self.window_size: int = config.replay_buffer_config.window_size
        self.action_space_size: int = config.game_config.action_space_size
        self.effective_discount: float = config.game_config.discount
        if config.game_config.num_players == 2:
            self.effective_discount *= -1
        self.num_unroll_steps: int = config.training_config.num_unroll_steps
        self.td_steps: int = config.training_config.td_steps

        self.buffer: List[GameHistory] = []
        self.sample_history: Callable[[], GameHistory] = self.uniform_sampler

        # For keeping track of statistics
        self.total_games: int = 0
        self.positions_in_buffer = tf.keras.metrics.Sum(name='Replay Buffer/Positions in buffer')
        self.average_total_value = tf.keras.metrics.Mean(name='Replay Buffer/Average total value')
        self.average_game_length = tf.keras.metrics.Mean(name='Replay Buffer/Average game length')
        self.average_num_batches = tf.keras.metrics.Mean(name='Replay Buffer/Average number of batches per history')
        self.networks: Dict[str, tf.keras.metrics.Metric] = {}
        self.agents: Dict[str, tf.keras.metrics.Metric] = {}

        self.metrics: List[tf.keras.metrics.Metric] = [self.positions_in_buffer,
                                                       self.average_total_value,
                                                       self.average_game_length,
                                                       self.average_num_batches]

        # Locks for gracefully handling multi-threading
        self.buffer_lock: RLock = RLock()
        self.networks_lock: RLock = RLock()
        self.agents_lock: RLock = RLock()

    def num_games(self) -> int:
        with self.buffer_lock:
            return len(self.buffer)

    def save_games(self, filepath: str) -> None:
        with self.buffer_lock:
            save_games(self.buffer, filepath)

    def load_games(self, filepath: str) -> None:
        for history in load_games(filepath):
            self.save_history(history)

    def compute_target_value(self, history: GameHistory, index: int) -> Value:
        bootstrap_index = index + self.td_steps
        if bootstrap_index < len(history):
            value = history.root_values[bootstrap_index] * self.effective_discount ** self.td_steps
        else:
            value = 0
        value += sum(reward * self.effective_discount ** i for i, reward in enumerate(history.rewards[index:bootstrap_index]))
        return Value(value)

    def preprocess_history(self, history: GameHistory) -> None:
        # Extend actions past the terminal state using random actions
        extended_actions = history.actions.copy()
        extended_actions.extend([Action(random.randrange(self.action_space_size)) for _ in range(self.num_unroll_steps)])
        history.extended_actions = tf.constant(extended_actions, dtype=tf.int32)

        # Extend target rewards past the terminal state using null rewards
        target_rewards = history.rewards.copy()
        target_rewards.extend([0 for _ in range(self.num_unroll_steps)])
        history.target_rewards = self.config.reward_config.inv_to_scalar(tf.constant(target_rewards, dtype=tf.float32))

        # Extend target values past the terminal state using the last value
        target_values = [self.compute_target_value(history, index) for index in range(len(history))]
        target_values.extend([0 for _ in range(self.num_unroll_steps+1)])
        history.target_values = self.config.value_config.inv_to_scalar(tf.constant(target_values, dtype=tf.float32))

        # Extend target policies past the terminal state using uniform policies
        history.target_policies = tf.concat(
            [
                tf.stack(history.policies),
                tf.ones(shape=(self.num_unroll_steps+1, self.action_space_size)) / self.action_space_size
            ], axis=0)

        history.total_value = sum(reward * self.effective_discount**i for i, reward in enumerate(history.rewards))
        history.metadata['num_batches'] = 0

    def save_history(self, game_history: GameHistory) -> None:
        self.preprocess_history(game_history)
        if self.num_games() < self.window_size:
            game_history_out = None
            with self.buffer_lock:
                self.buffer.append(game_history)
        else:
            position = self.total_games % self.window_size
            with self.buffer_lock:
                game_history_out = self.buffer[position]
                self.buffer[position] = game_history
        self.update_stats(game_history_in=game_history, game_history_out=game_history_out)

    def uniform_sampler(self) -> GameHistory:
        """
        Sample uniformly from all the games in the buffer.
        """
        with self.buffer_lock:
            return random.choice(self.buffer)

    @staticmethod
    def sample_position(game_history: GameHistory) -> int:
        """
        Sample uniformly from all the positions in a game.
        """
        return random.randrange(len(game_history)+1)

    def datapoint(self) -> Tuple[Tuple[Observation, ActionBatch], Tuple[ValueBatch, ValueBatch, PolicyBatch]]:
        history = self.sample_history()
        history.metadata['num_batches'] += 1

        game_pos = self.sample_position(history)
        observation = history.make_image(game_pos)
        actions = history.extended_actions[game_pos:game_pos + self.num_unroll_steps]
        target_rewards = history.target_rewards[game_pos:game_pos + self.num_unroll_steps]
        target_values = history.target_values[game_pos:game_pos + self.num_unroll_steps + 1]
        target_policies = history.target_policies[game_pos:game_pos + self.num_unroll_steps + 1]

        # actions.shape = (num_unroll_steps, )
        # target_rewards.shape = (num_unroll_steps+1, 1) or (num_unroll_steps+1, reward_support_size+1)
        # target_values.shape = (num_unroll_steps+1, 1) or (num_unroll_steps+1, value_support_size+1)
        # target_policies.shape = (num_unroll_steps+1, action_space_size)
        return (observation, actions), (target_rewards, target_values, target_policies)

    def as_dataset(self, batch_size: int) -> tf.data.Dataset:
        inputs, outputs = self.datapoint()
        inputs_spec = tuple(map(tf.TensorSpec.from_tensor, inputs))
        outputs_spec = tuple(map(tf.TensorSpec.from_tensor, outputs))

        dataset_signature = (inputs_spec, outputs_spec)
        dataset = tf.data.Dataset.from_generator(lambda: iter(self.datapoint, None), output_signature=dataset_signature)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=self.config.replay_buffer_config.prefetch_buffer_size)
        return dataset

    def update_stats(self, game_history_in: GameHistory, game_history_out: Optional[GameHistory] = None) -> None:
        if game_history_out:
            self.average_total_value(game_history_out.total_value, sample_weight=-1)
            self.average_game_length(len(game_history_out), sample_weight=-1)
            self.positions_in_buffer(-len(game_history_out))
            self.average_num_batches(game_history_out.metadata.get('num_batches', 0))

        self.average_total_value(game_history_in.total_value, sample_weight=1)
        self.average_game_length(len(game_history_in), sample_weight=1)
        self.positions_in_buffer(len(game_history_in))

        network = game_history_in.metadata.get('network_id', 'other')
        with self.networks_lock:
            network_metric = self.networks.setdefault(network, tf.keras.metrics.Mean(name=f'Networks/{network}'))
        network_metric(game_history_in.total_value)

        agent = game_history_in.metadata.get('agent_id', 'other')
        with self.agents_lock:
            agent_metric = self.agents.setdefault(agent, RollingMean(name=f'Agents/{agent}', window_size=1000))
        agent_metric(game_history_in.total_value)

        self.total_games += 1

    def stats(self) -> Dict[str, Any]:
        """
        Returns pre-computed replay buffer statistics.
        """
        with self.agents_lock:
            num_agents = len(self.agents)
        with self.networks_lock:
            num_networks = len(self.networks)
        stats = {
            'Replay Buffer/Total number of games': self.total_games,
            'Replay Buffer/Games in buffer': self.num_games(),
            'Replay Buffer/Number of agents': num_agents,
            'Replay Buffer/Number of networks': num_networks
        }
        stats.update({metric.name: metric.result().numpy() for metric in self.metrics})
        return stats

    def detailed_stats(self) -> Dict[str, Any]:
        """
        Returns more detailed replay buffer statistics.
        """
        stats = self.stats()
        with self.agents_lock:
            for metric in self.agents.values():
                played_games, average_value = metric.result().numpy()
                stats.update({metric.name+': games played': played_games,
                              metric.name+': average total value': average_value})
        with self.networks_lock:
            for metric in self.networks.values():
                stats.update({metric.name+': games played': metric.count.numpy(),
                              metric.name+': average total value': metric.result().numpy()})
        return stats
