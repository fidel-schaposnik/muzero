import os
import numpy as np
import tensorflow as tf

from muzero.config import MuZeroConfig
from muzero.utils import scalar_to_support

# For type annotations
from typing import List, Tuple, Callable, Iterable, Optional

from muzero.muprover_types import Value, Policy, Action, Observation, ValueBatch, PolicyBatch, ActionBatch, ObservationBatch


def one_hot_tensor_encoder(state_shape: Iterable[Optional[int]],
                           action_space_size: int,
                           name: str = 'OneHotTensorEncoder') -> tf.keras.Model:

    """
    Encode a batch of actions as one-hot tensors, then concatenate with hidden-state batch.

    Returns:
    
    Keras model that takes as an input:
      * hidden_state: a tensor of shape state_shape + [hidden_dimension]
      * action: an integer 

    and returns
      a tensor of shape state_shape + [hidden_dimension + action_space_size]
      such that output(*, *, ..., *, hidden_dimension + i) is 1 if i==action else 0
    """
    n = len(list(state_shape))
    hidden_state = tf.keras.Input(shape=list(state_shape) + [None], dtype=tf.float32, name='hidden_state')
    action = tf.keras.Input(shape=(), dtype=tf.int32, name='action')

    encoded_action_space = tf.transpose(tf.eye(action_space_size, batch_shape=state_shape),
                                        perm=[n] + list(range(n)) + [n+1])
    encoded_action = tf.gather(encoded_action_space, action)
    encoded_state_action = tf.keras.layers.concatenate([hidden_state, encoded_action], axis=-1)

    return tf.keras.Model(inputs=[hidden_state, action], outputs=encoded_state_action, name=name)


def tensor_product_encoder(
        hidden_state_shape: List[Optional[int]],
        action_space_size: int,
        name: str = 'TensorProductEncoder') -> tf.keras.Model:

    """ Encode a state and an action as a tensor product 
    
    Returns:

    Keras model that takes as an input:
      * hidden_state: a tensor of shape [batch_size] + hidden_state_shape
      * action: a tensor of shape [batch_size]

    and returns 
      * output: a tensor of shape [batch_size, action_space_size] + hidden_state_shape
    
    such that non-zero elements in the output are
      output[batch_index, actions[batch_index], *]  = hidden_states[batch_index, *]
    """

    hidden_state = tf.keras.Input(shape=hidden_state_shape, dtype=tf.float32, name='hidden_state')
    action = tf.keras.Input(shape=(), dtype=tf.int32, name='action')

    current_shape = tf.shape(hidden_state)
    batch_size = current_shape[0]
    
    indices = tf.stack([tf.range(batch_size), action], axis=1)
    
    encoded_action_state = tf.scatter_nd(indices=indices,
                                         updates=hidden_state,
                                         shape=[batch_size, action_space_size] + hidden_state_shape)
    return tf.keras.Model(inputs=[hidden_state, action], outputs=encoded_action_state, name=name)


def test_tensor_product_encoder():
    t = tensor_product_encoder(hidden_state_shape=[2, 2], action_space_size=3)
    states = tf.stack([tf.eye(2), 2*tf.eye(2), 3*tf.eye(2), 4*tf.eye(2)])
    actions = tf.constant([0, 2, 1, 2])
    result = t([states, actions])
    i = tf.eye(2)
    z = tf.zeros((2, 2))
    answer = tf.convert_to_tensor([[i, z, z], [z, z, 2*i], [z, 3*i, z], [z, z, 4*i]], dtype=tf.float32)
    assert tf.norm(answer - result) < 1e-7
    

def binary_plane_encoder(state_shape: Iterable[Optional[int]],
                         axis: int,
                         name: str = 'BinaryPlaneEncoder'
                         ) -> tf.keras.Model:
    """
    Encode actions as binary planes, append these to the hidden-state batch.
    """

    hidden_state = tf.keras.Input(shape=state_shape, dtype=tf.float32, name='hidden_state')
    action = tf.keras.Input(shape=(), dtype=tf.int32, name='action')

    plane_shape = [1 if i == axis else size for i, size in enumerate(state_shape)]
    planes = tf.stack([tf.zeros(shape=plane_shape), tf.ones(shape=plane_shape)])
    encoded_action = tf.gather(planes, action)
    encoded_state_action = tf.concat([hidden_state, encoded_action], axis=axis + 1)

    return tf.keras.Model(inputs=[hidden_state, action], outputs=encoded_state_action, name=name)


def scalar_to_support_model(input_shape: Iterable[Optional[int]],
                            scalar_min: tf.Tensor,
                            scalar_max: tf.Tensor,
                            support_size: int,
                            name: str = 'scalar_to_support'
                            ) -> tf.keras.Model:
    scalars = tf.keras.Input(shape=input_shape, name='scalars')
    normalized_scalars = (scalars - scalar_min)/(scalar_max - scalar_min)
    outputs = tf.keras.layers.Lambda(lambda x_scalar: scalar_to_support(x_scalar, support_size))(normalized_scalars)
    return tf.keras.Model(inputs=scalars, outputs=outputs, name=name)


def dummy_model(input_shape: Iterable[Optional[int]], name: str = 'no_op'):
    x = tf.keras.Input(shape=input_shape)
    return tf.keras.Model(inputs=x, outputs=x, name=name)


class NetworkOutput:
    def __init__(self,
                 hidden_state: Observation,
                 reward: Optional[Value],
                 value: Value,
                 policy_logits: Policy,
                 ) -> None:
        self.hidden_state: Observation = hidden_state
        self.reward: Optional[Value] = reward
        self.value: Value = value
        self.policy_logits: Policy = policy_logits

    def __repr__(self) -> str:
        return 'NetworkOutput(hidden_state={}, reward={}, value={}, policy_logits={})'.format(self.hidden_state,
                                                                                              self.reward,
                                                                                              self.value,
                                                                                              self.policy_logits)

    def masked_policy(self, actions: List[Action]) -> np.ndarray:
        policy_mask = np.zeros_like(self.policy_logits.numpy())
        policy_mask[actions] = 1
        policy = np.exp(self.policy_logits.numpy()) * policy_mask
        return policy / policy.sum()


class BatchNetworkOutput:
    def __init__(self,
                 batch_hidden_state: ObservationBatch,
                 batch_reward: Optional[ValueBatch],
                 batch_value: ValueBatch,
                 batch_policy_logits: PolicyBatch,
                 reward_to_scalar: Callable[[tf.Tensor], tf.Tensor],
                 value_to_scalar: Callable[[tf.Tensor], tf.Tensor]
                 ) -> None:
        self.batch_hidden_state: ObservationBatch = batch_hidden_state
        self.batch_reward: Optional[ValueBatch] = batch_reward
        self.batch_value: ValueBatch = batch_value
        self.batch_policy_logits: PolicyBatch = batch_policy_logits
        self.reward_to_scalar: Callable[[tf.Tensor], tf.Tensor] = reward_to_scalar
        self.value_to_scalar: Callable[[tf.Tensor], tf.Tensor] = value_to_scalar

    def split_batch(self) -> List[NetworkOutput]:
        batch_reward_scalar = self.reward_to_scalar(self.batch_reward) if self.batch_reward is not None else iter(lambda: None, 0)
        batch_value_scalar = self.value_to_scalar(self.batch_value)
        return [NetworkOutput(hidden_state=Observation(hidden_state),
                              reward=Value(float(reward_scalar)) if reward_scalar is not None else None,
                              value=Value(float(value_scalar)),
                              policy_logits=policy_logits
                              )
                for hidden_state, reward_scalar, value_scalar, policy_logits in
                zip(self.batch_hidden_state, batch_reward_scalar, batch_value_scalar, self.batch_policy_logits)]


class Network:
    """
    Base class for all of MuZero neural networks.
    """
    def __init__(self,
                 config: MuZeroConfig,
                 representation: tf.keras.Model,
                 dynamics: tf.keras.Model,
                 prediction: tf.keras.Model,
                 state_preprocessing: tf.keras.Model,
                 state_action_encoder: Callable[[Tuple[ObservationBatch, ActionBatch]], ObservationBatch],
                 ) -> None:
        self.config = config
        self.representation: tf.keras.Model = representation
        self.dynamics: tf.keras.Model = dynamics
        self.prediction: tf.keras.Model = prediction
        self.state_preprocessing: tf.keras.Model = state_preprocessing
        self.state_action_encoder: Callable[[Tuple[ObservationBatch, ActionBatch]], ObservationBatch] = state_action_encoder

        self.steps: tf.Variable = tf.Variable(initial_value=0, dtype=tf.int32, name='training_steps')
        self.checkpoint: tf.train.Checkpoint = tf.train.Checkpoint(steps=self.steps,
                                                                   representation=self.representation,
                                                                   dynamics=self.dynamics,
                                                                   prediction=self.prediction
                                                                   )

        self.initial_inference_model: tf.keras.Model = self.build_initial_inference_model()
        self.recurrent_inference_model: tf.keras.Model = self.build_recurrent_inference_model()

    def training_steps(self) -> int:
        return self.steps.numpy()

    def build_initial_inference_model(self) -> tf.keras.Model:
        observation = tf.keras.Input(shape=self.state_preprocessing.input_shape[1:], dtype=tf.float32,
                                     name=self.config.network_config.OBSERVATION)

        hidden_state = self.representation(self.state_preprocessing(observation))

        value, policy_logits = self.prediction(hidden_state)

        return tf.keras.Model(inputs=observation,
                              outputs=[hidden_state, value, policy_logits],
                              name=self.config.network_config.INITIAL_INFERENCE)

    def initial_inference(self, observation: ObservationBatch) -> BatchNetworkOutput:
        hidden_state, value, policy_logits = self.initial_inference_model(observation, training=False)

        return BatchNetworkOutput(batch_hidden_state=ObservationBatch(hidden_state),
                                  batch_reward=None,
                                  batch_value=ValueBatch(value),
                                  batch_policy_logits=PolicyBatch(policy_logits),
                                  reward_to_scalar=self.config.reward_config.to_scalar,
                                  value_to_scalar=self.config.value_config.to_scalar)

    def build_recurrent_inference_model(self) -> tf.keras.Model:
        hidden_state = tf.keras.Input(shape=self.representation.output_shape[1:], dtype=tf.float32,
                                      name=self.config.network_config.HIDDEN_STATE)
        action = tf.keras.Input(shape=(), dtype=tf.int32, name=self.config.network_config.ACTION)

        dynamics_input = self.state_action_encoder((hidden_state, action))

        new_hidden_state, reward = self.dynamics(dynamics_input)

        value, policy_logits = self.prediction(new_hidden_state)

        return tf.keras.Model(inputs=[hidden_state, action],
                              outputs=[new_hidden_state, reward, value, policy_logits],
                              name=self.config.network_config.RECURRENT_INFERENCE)

    def recurrent_inference(self, hidden_state: ObservationBatch, action: ActionBatch) -> BatchNetworkOutput:
        hidden_state, reward, value, policy_logits = self.recurrent_inference_model((hidden_state, action), training=False)

        return BatchNetworkOutput(batch_hidden_state=ObservationBatch(hidden_state),
                                  batch_reward=ValueBatch(reward),
                                  batch_value=ValueBatch(value),
                                  batch_policy_logits=PolicyBatch(policy_logits),
                                  reward_to_scalar=self.config.reward_config.to_scalar,
                                  value_to_scalar=self.config.value_config.to_scalar)

    def save(self, checkpoint_prefix: str) -> None:
        self.checkpoint.write(checkpoint_prefix)

    def load(self, checkpoint_prefix: str) -> None:
        self.checkpoint.read(checkpoint_prefix)

    def save_tfx_models(self, saved_models_path: str) -> None:
        self.initial_inference_model.save(
            os.path.join(saved_models_path, self.config.network_config.INITIAL_INFERENCE, str(self.training_steps())))
        self.recurrent_inference_model.save(
            os.path.join(saved_models_path, self.config.network_config.RECURRENT_INFERENCE, str(self.training_steps())))

    def summary(self) -> None:
        self.representation.summary()
        self.dynamics.summary()
        self.prediction.summary()

    def __repr__(self) -> str:
        return 'Network({}, training_steps={})'.format(
            {model.name: {'parameters': model.count_params(),
                          'input_shape': model.input_shape,
                          'output_shape': model.output_shape}
             for model in [self.representation, self.dynamics, self.prediction]},
            self.training_steps())
