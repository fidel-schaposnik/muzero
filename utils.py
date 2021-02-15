import random
import pickle
import string
import datetime
import argparse
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from collections import namedtuple
from importlib import import_module


# For type annotations
from typing import Optional, Dict, Any

from muzero.muprover_types import Value


MAXIMUM_FLOAT_VALUE = float('inf')
KnownBounds = namedtuple('KnownBounds', ['min', 'max'])


def disable_gpu():
    """ disables GPU visibility in tensorflow
    """
    tf.config.set_visible_devices([], "GPU")
    logical_devices = tf.config.list_logical_devices("GPU")
    print("TF logical devices", logical_devices)


def timestamp() -> str:
    return datetime.datetime.now().strftime("%d-%m-%Y--%H-%M")


def random_id() -> str:
    return ''.join(random.choice(string.ascii_letters) for _ in range(8))


def to_bytes_dict(dictionary: Dict[str, Any]) -> Dict[str, bytes]:
    return {key: pickle.dumps(value) for key, value in dictionary.items()}


def from_bytes_dict(dictionary: Dict[str, bytes]) -> Dict[str, Any]:
    return {key: pickle.loads(value) for key, value in dictionary.items()}


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: Value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: Value) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class CommandLineParser(argparse.ArgumentParser):
    def __init__(self,
                 name: str,
                 game: bool = False,
                 environment: bool = False,
                 network: bool = False,
                 replay_buffer: bool = False,
                 port: bool = False,
                 threads: bool = False
                 ) -> None:
        super().__init__(description=name)
        self.game = game
        self.environment = environment
        self.network = network
        self.replay_buffer = replay_buffer
        self.port = port
        self.threads = threads

        if self.game:
            self.add_argument('--game', type=str, required=True,
                              help='One of the games implemented in the games/ directory')
        if self.environment:
            self.add_argument('--environment', type=str, metavar='IP:PORT', required=True,
                              help='ip:port for gRPC communication with an environment server')
        if self.network:
            self.add_argument('--network', type=str, metavar='IP:PORT', required=True,
                              help='ip:port for gRPC communication with a network server')
        if self.replay_buffer:
            self.add_argument('--replay_buffer', type=str, metavar='IP:PORT', required=True,
                              help='ip:port for gRPC communication with a replay buffer server')
        if self.port:
            self.add_argument('--port', type=str, required=True,
                              help='Port for gRPC communication')
        if self.threads:
            self.add_argument('--threads', type=int, default=10,
                              help='Number of threads for the gRPC server')

        # Common arguments (available for all services)
        self.add_argument('--disable-gpu', action='store_true', help='Disable GPU usage by TensorFlow in this process')

    def parse_args(self, args=None, namespace=None) -> argparse.Namespace:
        args = super().parse_args(args=args, namespace=namespace)

        if self.threads and args.threads < 1:
            self.error(f'Number of --threads for the gRPC server should be strictly positive!')

        if args.disable_gpu:
            disable_gpu()
            
        if self.game:
            try:
                game_module = import_module(f'muzero.games.{args.game}')
            except ModuleNotFoundError:
                self.error(f'Choice of --game {args.game} is invalid!')
            else:
                args.config = game_module.make_config()
        return args


def scalar_to_support(x_scalar: tf.Tensor, support_size: int) -> tf.Tensor:
    y = support_size * (tf.math.sqrt(x_scalar + tf.ones_like(x_scalar)) - 1) / (tf.math.sqrt(2.) - 1)
    low = tf.math.floor(y)
    high = tf.cast(tf.math.ceil(y), dtype=tf.int32)
    p = tf.expand_dims(y - low, axis=-1)
    return p * tf.one_hot(high, support_size + 1) + (1-p) * tf.one_hot(tf.cast(low, dtype=tf.int32), support_size + 1)


def support_to_scalar(x_conv: tf.Tensor, support_size: int) -> tf.Tensor:
    y_list = tf.math.reduce_sum(x_conv * tf.range(support_size+1, dtype=tf.float32), axis=-1)/support_size
    return ((tf.sqrt(2.)-1) * y_list + 1)**2 - 1


class RollingMean(tf.keras.metrics.Metric):
    def __init__(self, name: str, window_size: int) -> None:
        super().__init__(name=name)
        self.window_size = self.add_weight(name='window_size', dtype=tf.int32)
        self.values = self.add_weight(name='values', shape=(window_size,), dtype=tf.float32, initializer='zeros')
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer='zeros')

        self.window_size.assign(window_size)

    def update_state(self, value: float):
        position = tf.math.mod(self.count, self.window_size)
        old = self.values * tf.one_hot(position, self.window_size)
        new = value * tf.one_hot(position, self.window_size)
        self.values.assign_add(new - old)
        self.count.assign_add(1)

    def result(self):
        num_points = tf.cast(tf.math.minimum(self.count, self.window_size), dtype=tf.float32)
        return tf.cast(self.count, dtype=tf.float32), tf.math.reduce_sum(self.values) / num_points

    def reset_states(self):
        self.values.assign(tf.zeros(self.window_size))
        self.count.assign(0)


def hparams_safe_update(hyperparameters, parameter_dict):
    for key, value in parameter_dict.items():
        try:
            hp.hparams({key: value})
        except TypeError:
            continue
        else:
            hyperparameters[key] = value
