import random
import string
import datetime
from importlib import import_module

MAXIMUM_FLOAT_VALUE = float('inf')
NOT_A_NUMBER = float('nan')


def load_game(game_name, parser):
    try:
        game_module = import_module('games.{}'.format(game_name))
    except ModuleNotFoundError:
        parser.error('Choice of --game {} is invalid!'.format(game_name))
    else:
        return game_module.make_config()


def timestamp():
    return datetime.datetime.now().strftime("%d-%m-%Y--%H-%M")


def random_id():
    return ''.join(random.choice(string.ascii_letters) for _ in range(8))


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds):
        self.minimum, self.maximum = known_bounds if known_bounds else MAXIMUM_FLOAT_VALUE, -MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

    def __str__(self):
        return 'MinMaxStats(min={:.6f}, max={:.6f})'.format(self.minimum, self.maximum)
