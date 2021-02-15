import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from config import MuZeroConfig
from network import BatchNetworkOutput

# For type annotations
from muzero_types import ActionBatch, ObservationBatch, PolicyBatch, ValueBatch


class RemoteNetwork:
    def __init__(self, config: MuZeroConfig, ip_port: str, timeout: float = 5.0) -> None:
        self.config = config
        channel = grpc.insecure_channel(ip_port)
        self.prediction_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        self.timeout: float = timeout
        self.steps: int = -1

        # Network input names
        self.OBSERVATION = config.network_config.OBSERVATION
        self.HIDDEN_STATE = config.network_config.HIDDEN_STATE
        self.ACTION = config.network_config.ACTION

        # Network output names
        self.REPRESENTATION_HIDDEN_STATE = config.network_config.REPRESENTATION
        self.DYNAMICS_HIDDEN_STATE = config.network_config.DYNAMICS
        self.DYNAMICS_REWARD = f'{config.network_config.DYNAMICS}_1'
        self.PREDICTION_VALUE = config.network_config.PREDICTION
        self.PREDICTION_POLICY_LOGITS = f'{config.network_config.PREDICTION}_1'

    def initial_inference(self, observation: ObservationBatch) -> BatchNetworkOutput:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.config.network_config.INITIAL_INFERENCE
        request.inputs[self.OBSERVATION].CopyFrom(tf.make_tensor_proto(observation))
        response = self.prediction_service.Predict(request, self.timeout)
        self.steps = response.model_spec.version.value

        hidden_state = tf.constant(tf.make_ndarray(response.outputs[self.REPRESENTATION_HIDDEN_STATE]))
        value = tf.constant(tf.make_ndarray(response.outputs[self.PREDICTION_VALUE]))
        policy_logits = tf.constant(tf.make_ndarray(response.outputs[self.PREDICTION_POLICY_LOGITS]))

        return BatchNetworkOutput(batch_hidden_state=ObservationBatch(hidden_state),
                                  batch_reward=None,
                                  batch_value=ValueBatch(value),
                                  batch_policy_logits=PolicyBatch(policy_logits),
                                  reward_to_scalar=self.config.reward_config.to_scalar,
                                  value_to_scalar=self.config.value_config.to_scalar)

    def recurrent_inference(self, hidden_state: ObservationBatch, action: ActionBatch) -> BatchNetworkOutput:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.config.network_config.RECURRENT_INFERENCE
        request.inputs[self.HIDDEN_STATE].CopyFrom(tf.make_tensor_proto(hidden_state))
        request.inputs[self.ACTION].CopyFrom(tf.make_tensor_proto(action))
        response = self.prediction_service.Predict(request, self.timeout)
        self.steps = response.model_spec.version.value

        hidden_state = tf.constant(tf.make_ndarray(response.outputs[self.DYNAMICS_HIDDEN_STATE]))
        reward = tf.constant(tf.make_ndarray(response.outputs[self.DYNAMICS_REWARD]))
        value = tf.constant(tf.make_ndarray(response.outputs[self.PREDICTION_VALUE]))
        policy_logits = tf.constant(tf.make_ndarray(response.outputs[self.PREDICTION_POLICY_LOGITS]))

        return BatchNetworkOutput(batch_hidden_state=ObservationBatch(hidden_state),
                                  batch_reward=ValueBatch(reward),
                                  batch_value=ValueBatch(value),
                                  batch_policy_logits=PolicyBatch(policy_logits),
                                  reward_to_scalar=self.config.reward_config.to_scalar,
                                  value_to_scalar=self.config.value_config.to_scalar)

    def training_steps(self) -> int:
        return self.steps
