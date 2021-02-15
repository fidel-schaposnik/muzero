from typing import NewType, NamedTuple, Optional, List
import tensorflow as tf

Value = NewType('Value', float)

Observation = NewType('Observation', tf.Tensor)
Player = NewType('Player', int)
Action = NewType('Action', int)

class State(NamedTuple):
    observation: Observation
    to_play: Player
    legal_actions: List[Action]

Policy = NewType('Policy', tf.Tensor)
ObservationBatch = NewType('ObservationBatch', tf.Tensor)
ActionBatch = NewType('ActionBatch', tf.Tensor)
ValueBatch = NewType('ValueBatch', tf.Tensor)
PolicyBatch = NewType('PolicyBatch', tf.Tensor)
