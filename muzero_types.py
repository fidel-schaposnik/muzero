from typing import NewType
import tensorflow as tf


State = NewType('State', tf.Tensor)
Observation = NewType('Observation', tf.Tensor)
Value = NewType('Value', float)
Action = NewType('Action', int)
Policy = NewType('Policy', tf.Tensor)
ObservationBatch = NewType('ObservationBatch', tf.Tensor)
ActionBatch = NewType('ActionBatch', tf.Tensor)
ValueBatch = NewType('ValueBatch', tf.Tensor)
PolicyBatch = NewType('PolicyBatch', tf.Tensor)
