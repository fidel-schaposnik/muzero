import tensorflow as tf
import numpy as np
import datetime, string, random
from math import isnan


MAXIMUM_FLOAT_VALUE = float('inf')
NOT_A_NUMBER = float('nan')


def timestamp():
    return datetime.datetime.now().strftime("%d-%m-%Y--%H-%M")


def random_id():
    return ''.join(random.choice(string.ascii_letters) for _ in range(8))


class MinMaxStats:
    """
    A class to normalize values and perform scalar <--> categorical transformations.
    """

    def __init__(self, known_bounds=None):
        self.minimum, self.maximum = known_bounds if known_bounds else (MAXIMUM_FLOAT_VALUE, -MAXIMUM_FLOAT_VALUE)

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return 0.5 if isnan(value) else (value - self.minimum) / (self.maximum - self.minimum)
        else:
            return 0.0 if isnan(value) else value

    def __str__(self):
        return 'MinMaxStats(min={:.6f}, max={:.6f})'.format(self.minimum, self.maximum)


def scalar_to_support(x_scalar, support_size):
    num_x = len(x_scalar)
    y = support_size * (np.sqrt(x_scalar + 1) - 1) / (np.sqrt(2) - 1)
    low = np.floor(y).astype(np.int)
    high = np.ceil(y).astype(np.int)
    p = y - low
    result = np.zeros((num_x, support_size + 1), dtype=np.float32)
    result[range(num_x), high] = p
    result[range(num_x), low] = 1 - p
    return result


def support_to_scalar(x_conv, support_size):
    y_list = tf.math.reduce_sum(x_conv * np.arange(support_size+1), axis=-1)/support_size
    return ((np.sqrt(2)-1) * y_list + 1)**2 -1
