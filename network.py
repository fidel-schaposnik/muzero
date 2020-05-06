from environment import *
from utils import *
import time, os, grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import get_model_metadata_pb2


def residual_block(name, inputs, num_filters, kernel_size):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size, padding='same', name=name + '_conv1')(inputs)
    x = tf.keras.layers.BatchNormalization(name=name + '_norm1')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_relu1')(x)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size, padding='same', name=name + '_conv2')(x)
    x = tf.keras.layers.BatchNormalization(name=name + '_norm2')(x)
    x = tf.keras.layers.Add(name=name + '_skip')([x, inputs])
    x = tf.keras.layers.Activation('relu', name=name + '_relu2')(x)
    return x


def residual_tower(name, inputs, num_filters, kernel_size, height):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size, padding='same', name=name + '_first_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(name=name + '_first_norm')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_first_relu')(x)
    for h in range(height):
        x = residual_block(name=name + '_res' + str(h), inputs=x, num_filters=num_filters, kernel_size=kernel_size)
    return x


def logits_head(name, inputs, num_logits, num_filters, kernel_size):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size, padding='same', name=name + '_conv')(inputs)
    # x = tf.keras.layers.BatchNormalization(name=name + '_norm')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_relu')(x)
    x = tf.keras.layers.Flatten(name=name + '_flatten')(x)
    x = tf.keras.layers.Dense(num_logits, name=name + '_dense', activation='relu')(x)
    return x


def scalar_head(name, inputs, scalar_activation, num_outputs, num_filters, kernel_size, hidden_size):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size, padding='same', name=name + '_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(name=name + '_norm')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_relu')(x)
    x = tf.keras.layers.Flatten(name=name + '_flatten')(x)
    x = tf.keras.layers.Dense(hidden_size, activation='relu', name=name + '_hidden')(x)
    x = tf.keras.layers.Dense(num_outputs, activation=scalar_activation, name=name + '_output')(x)
    return x


def categorical_scalar_head(name, inputs, num_outputs, scalar_support_size, num_filters, kernel_size, hidden_size):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size, padding='same', name=name + '_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(name=name + '_norm')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_relu')(x)
    x = tf.keras.layers.Flatten(name=name + '_flatten')(x)
    x = tf.keras.layers.Dense(num_outputs * hidden_size, activation='relu', name=name + '_hidden')(x)
    x = tf.keras.layers.Reshape(name=name + '_reshape', target_shape=(num_outputs, hidden_size))(x)
    x = tf.keras.layers.Dense(scalar_support_size+1, activation='softmax', name=name + '_output')(x)
    return x


def hidden_state_normalization(tensor):
    a = tf.math.reduce_min(tensor, axis=(1, 2, 3), keepdims=True)
    b = tf.math.reduce_max(tensor, axis=(1, 2, 3), keepdims=True)
    return (tensor - a) / tf.where(a == b, 1., b - a)


def dummy_network(name, input_shape, conv_filters):
    inputs = tf.keras.Input(shape=input_shape, name=name+'_input_image')
    outputs = tf.keras.layers.Lambda(lambda tensor: tf.pad(tensor, paddings=[(0,0), (0,0), (0,0), (0,conv_filters-input_shape[-1])]), input_shape=input_shape, name=name+'_lambda')(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def fully_connected_network(name, inputs, num_layers, num_units):
    x = tf.keras.layers.Flatten(name=name+'_flatten')(inputs)
    for i in range(num_layers):
        x = tf.keras.layers.Dense(units=num_units, activation='relu', name=name+'_dense'+str(i))(x)
    return x


def prediction_network(name, input_shape, num_logits, num_players,
                       tower_height, conv_filters, conv_kernel_size,
                       policy_filters, policy_kernel_size,
                       scalar_activation, value_filters, value_kernel_size, value_hidden_size):

    inputs = tf.keras.Input(shape=input_shape, name=name + '_input_hidden_state')

    residual_output = residual_tower(name=name + '_tower', inputs=inputs,
                                     height=tower_height, num_filters=conv_filters, kernel_size=conv_kernel_size)

    policy_output = logits_head(name=name + '_policy', inputs=residual_output, num_logits=num_logits,
                                num_filters=policy_filters, kernel_size=policy_kernel_size)

    value_output = scalar_head(name=name + '_value', inputs=residual_output,
                               num_outputs=num_players, scalar_activation=scalar_activation,
                               num_filters=value_filters, kernel_size=value_kernel_size, hidden_size=value_hidden_size)

    return tf.keras.Model(inputs=inputs, outputs=[policy_output, value_output], name=name)


def categorical_prediction_network(
        name, input_shape,
        tower_height, conv_filters, conv_kernel_size,
        num_logits, policy_filters, policy_kernel_size,
        num_players, scalar_support_size, value_filters, value_kernel_size, value_hidden_size):

    inputs = tf.keras.Input(shape=input_shape, name=name + '_input_hidden_state')

    residual_output = residual_tower(name=name + '_tower', inputs=inputs,
                                     height=tower_height, num_filters=conv_filters, kernel_size=conv_kernel_size)
    hidden_state = tf.keras.layers.Lambda(hidden_state_normalization, name=name + '_hidden_state_norm')(residual_output)

    policy_output = logits_head(name=name + '_policy', inputs=hidden_state,
                                num_logits=num_logits, num_filters=policy_filters, kernel_size=policy_kernel_size)

    value_output = categorical_scalar_head(name=name + '_value', inputs=hidden_state,
                                           num_outputs=num_players, scalar_support_size=scalar_support_size,
                                           num_filters=value_filters, kernel_size=value_kernel_size, hidden_size=value_hidden_size)

    return tf.keras.Model(inputs=inputs, outputs=[policy_output, value_output], name=name)


def fully_connected_prediction_network(name, input_shape, num_players, num_logits,
                                       num_layers, num_units,
                                       scalar_activation, value_filters, value_kernel_size, value_hidden_size,
                                       policy_filters, policy_kernel_size):
    inputs = tf.keras.Input(shape=input_shape, name=name + '_input_hidden_state')

    fcn_output = fully_connected_network(name=name+'_fcn', inputs=inputs, num_layers=num_layers, num_units=num_units)

    hidden_state = tf.keras.layers.Reshape(target_shape=(num_units, 1, 1), name=name + '_hidden_state')(fcn_output)

    policy_output = logits_head(name=name + '_policy', inputs=hidden_state, num_logits=num_logits,
                                num_filters=policy_filters, kernel_size=policy_kernel_size)

    value_output = scalar_head(name=name + '_value', inputs=hidden_state,
                               num_outputs=num_players, scalar_activation=scalar_activation,
                               num_filters=value_filters, kernel_size=value_kernel_size, hidden_size=value_hidden_size)

    return tf.keras.Model(inputs=inputs, outputs=[policy_output, value_output], name=name)


def dynamics_network(name, input_shape, num_players,
                     tower_height, conv_filters, conv_kernel_size,
                     scalar_activation, reward_filters, reward_kernel_size, reward_hidden_size,
                     toplay_filters, toplay_kernel_size
                     ):

    inputs = tf.keras.Input(shape=input_shape, name=name + '_input_hidden_state')

    residual_output = residual_tower(name=name + '_tower', inputs=inputs,
                                     height=tower_height, num_filters=conv_filters, kernel_size=conv_kernel_size)
    hidden_state = tf.keras.layers.Lambda(hidden_state_normalization, name=name + '_hidden_state_norm')(residual_output)

    reward_output = scalar_head(name=name + '_reward', inputs=hidden_state,
                                num_outputs=num_players, scalar_activation=scalar_activation,
                                num_filters=reward_filters, kernel_size=reward_kernel_size, hidden_size=reward_hidden_size)

    toplay_logits = logits_head(name=name + '_toplay', inputs=hidden_state, num_logits=num_players,
                                num_filters=toplay_filters, kernel_size=toplay_kernel_size)

    return tf.keras.Model(inputs=inputs, outputs=[hidden_state, reward_output, toplay_logits], name=name)


def categorical_dynamics_network(name, input_shape, num_players,
                                 tower_height, conv_filters, conv_kernel_size,
                                 scalar_support_size, reward_filters, reward_kernel_size, reward_hidden_size,
                                 toplay_filters, toplay_kernel_size
                                 ):

    inputs = tf.keras.Input(shape=input_shape, name=name + '_input_hidden_state')

    residual_output = residual_tower(name=name + '_tower', inputs=inputs,
                                     height=tower_height, num_filters=conv_filters, kernel_size=conv_kernel_size)
    hidden_state = tf.keras.layers.Lambda(hidden_state_normalization, name=name + '_hidden_state_norm')(residual_output)

    reward_output = categorical_scalar_head(name=name + '_reward', inputs=hidden_state, num_outputs=num_players,
                                            scalar_support_size=scalar_support_size, num_filters=reward_filters,
                                            kernel_size=reward_kernel_size, hidden_size=reward_hidden_size)

    toplay_logits = logits_head(name=name + '_toplay', inputs=hidden_state, num_logits=num_players,
                                num_filters=toplay_filters, kernel_size=toplay_kernel_size)

    return tf.keras.Model(inputs=inputs, outputs=[hidden_state, reward_output, toplay_logits], name=name)


def fully_connected_dynamics_network(name, input_shape, num_players,
                                     num_layers, num_units,
                                     scalar_activation, reward_filters, reward_kernel_size, reward_hidden_size,
                                     toplay_filters, toplay_kernel_size
                                     ):
    inputs = tf.keras.Input(shape=input_shape, name=name + '_input_hidden_state')

    hidden_state = fully_connected_network(name=name+'_fcn', inputs=inputs, num_layers=num_layers, num_units=num_units)

    hidden_state = tf.keras.layers.Reshape(target_shape=(num_units, 1, 1), name=name+'_hidden_state')(hidden_state)

    reward_output = scalar_head(name=name + '_reward', inputs=hidden_state,
                                num_outputs=num_players, scalar_activation=scalar_activation,
                                num_filters=reward_filters, kernel_size=reward_kernel_size,
                                hidden_size=reward_hidden_size)

    toplay_logits = logits_head(name=name + '_toplay', inputs=hidden_state, num_logits=num_players,
                                num_filters=toplay_filters, kernel_size=toplay_kernel_size)
    return tf.keras.Model(inputs=inputs, outputs=[hidden_state, reward_output, toplay_logits], name=name)


def representation_network(name, input_shape, conv_filters, conv_kernel_size, tower_height):
    inputs = tf.keras.Input(shape=input_shape, name=name + '_input_image')
    residual_output = residual_tower(name=name + '_tower', inputs=inputs,
                                     height=tower_height, num_filters=conv_filters, kernel_size=conv_kernel_size)
    hidden_state = tf.keras.layers.Lambda(hidden_state_normalization, name=name + '_hidden_state_norm')(residual_output)
    return tf.keras.Model(inputs=inputs, outputs=hidden_state, name=name)


def fully_connected_representation_network(name, input_shape, num_layers, num_units):
    inputs = tf.keras.Input(shape=input_shape, name=name + '_input_image')
    hidden_state = fully_connected_network(name=name+'_fcn', inputs=inputs, num_layers=num_layers, num_units=num_units)
    outputs = tf.keras.layers.Reshape(target_shape=(num_units, 1, 1), name=name + '_hidden_state')(hidden_state)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


class OneHotPlaneEncoder:
    """
    Encode a batch of actions as one-hot planes, then concatenate with hidden-state batch.
    """
    def __init__(self, rows, cols, action_space_size):
        self.encoded_action_space = np.zeros((action_space_size, rows, cols, action_space_size)).astype(np.float32)
        for i in range(action_space_size):
            self.encoded_action_space[i, :, :, i] = 1

    def __call__(self, batch_hidden_state, batch_action):
        batch_encoded_action = self.encoded_action_space[[action.index for action in batch_action]]
        batch_dynamics_input = tf.concat([batch_hidden_state, batch_encoded_action], axis=-1)
        return batch_dynamics_input


class BinaryPlaneEncoder:
    """
    Encode actions as binary planes, append these to the hidden-state batch.
    """
    def __call__(self, batch_hidden_state, batch_action):
        # Encode action as binary planes
        batch_encoded_action = np.zeros_like(batch_hidden_state)
        for i, action in enumerate(batch_action):
            batch_encoded_action[i] = action.index

        # Concatenate action to hidden state
        batch_dynamics_input = tf.concat([batch_hidden_state, batch_encoded_action], axis=-1)
        return batch_dynamics_input


class NetworkOutput:
    def __init__(self, value, reward, policy_logits, hidden_state, to_play, inv_scalar_transformation):
        self.value = value
        self.reward = reward
        self.policy_logits = policy_logits
        self.hidden_state = hidden_state
        self.to_play = to_play
        self.inv_scalar_transformation = inv_scalar_transformation

    def split_batch(self):
        return [NetworkOutput(
                              {Player(i): self.inv_scalar_transformation(value) for i, value in enumerate(values)},
                              {Player(i): self.inv_scalar_transformation(reward) for i, reward in enumerate(rewards)},
                              {Action(i): logit for i, logit in enumerate(policy_logits)},
                              hidden_state,
                              Player(tf.math.argmax(to_play)),
                              lambda x: x
                              )
                for values, rewards, policy_logits, hidden_state, to_play in
                zip(self.value, self.reward, self.policy_logits, self.hidden_state, self.to_play)
                ]


class Network:
    """
    Base class for all of MuZero neural networks.
    """

    def __init__(self, state_action_encoder):
        self.steps = 0
        self.representation = None
        self.dynamics = None
        self.prediction = None
        self.state_action_encoder = state_action_encoder
        self.inv_scalar_transformation = lambda x: x

        self.REPRESENTATION_TIME = 0.
        self.PREDICTION_TIME = 0.
        self.DYNAMICS_TIME = 0.
        self.ENCODING_TIME = 0.

    def training_steps(self):
        """
        Returns the number of steps this network has been trained for.
        """
        return self.steps

    def get_weights(self):
        """
        Returns the network weights.
        """
        weights = []
        for sub_network in [self.representation, self.dynamics, self.prediction]:
            weights.extend(sub_network.trainable_weights)
        return weights

    def save_weights(self, filepath_prefix):
        """
        Exports the network weights to HDF5 format.
        """
        self.representation.save_weights(filepath_prefix + '_rep.h5')
        self.dynamics.save_weights(filepath_prefix + '_dyn.h5')
        self.prediction.save_weights(filepath_prefix + '_pre.h5')

    def load_weights(self, filepath_prefix):
        """
        Loads the network weights from HDF5 format files.
        """
        self.representation.load_weights(filepath_prefix + '_rep.h5')
        self.dynamics.load_weights(filepath_prefix + '_dyn.h5')
        self.prediction.load_weights(filepath_prefix + '_pre.h5')

    def save_model(self, filepath_prefix):
        """
        Export the network models to SavedModel format.
        """
        tf.saved_model.save(self.representation, os.path.join(filepath_prefix, 'representation', str(self.steps)))
        tf.saved_model.save(self.dynamics, os.path.join(filepath_prefix, 'dynamics', str(self.steps)))
        tf.saved_model.save(self.prediction, os.path.join(filepath_prefix, 'prediction', str(self.steps)))

    def summary(self):
        """
        Print summaries for all the neural networks we use.
        """
        self.representation.summary()
        self.dynamics.summary()
        self.prediction.summary()

    def hidden_state_shape(self, batch_size=None):
        """
        Returns the shape of a batch of hidden states with the current network parameters.
        """
        raise ImplementationError('hidden_state_shape', 'Network')

    def toplay_shape(self, batch_size=None):
        raise ImplementationError('toplay_shape', 'Network')

    def initial_inference(self, batch_image, training=False):
        """
        Apply representation + prediction networks to a batch of observations.
        """
        start = time.time()
        batch_hidden_state = self.representation(batch_image, training=training)
        end = time.time()
        self.REPRESENTATION_TIME += end-start

        start = time.time()
        batch_policy_logits, batch_value = self.prediction(batch_hidden_state, training=training)
        end = time.time()
        self.PREDICTION_TIME += end - start

        return NetworkOutput(value=batch_value,
                             reward=tf.zeros_like(batch_value),
                             policy_logits=batch_policy_logits,
                             hidden_state=batch_hidden_state,
                             to_play=tf.zeros(self.toplay_shape(len(batch_image))),
                             inv_scalar_transformation=self.inv_scalar_transformation)

    def recurrent_inference(self, batch_hidden_state, batch_action, training=False):
        """
        Apply the dynamics + prediction networks to a batch of hidden states.
        """
        start = time.time()
        batch_dynamics_input = self.state_action_encoder(batch_hidden_state, batch_action)
        end = time.time()
        self.ENCODING_TIME += end-start

        start = time.time()
        batch_hidden_state, batch_reward, batch_toplay = self.dynamics(batch_dynamics_input, training=training)
        end = time.time()
        self.DYNAMICS_TIME += end-start

        start = time.time()
        batch_policy_logits, batch_value = self.prediction(batch_hidden_state, training=training)
        end = time.time()
        self.PREDICTION_TIME += end-start

        return NetworkOutput(value=batch_value,
                             reward=batch_reward,
                             policy_logits=batch_policy_logits,
                             hidden_state=batch_hidden_state,
                             to_play=batch_toplay,
                             inv_scalar_transformation=self.inv_scalar_transformation)

    def __str__(self):
        return 'Network({}, trining_steps={})'.format(
            {model.name: {'parameters': model.count_params(),
                          'input_shape': model.input_shape,
                          'output_shape': model.output_shape}
             for model in [self.representation, self.dynamics, self.prediction]},
            self.training_steps())


class RemoteNetwork(Network):
    def __init__(self, state_action_encoder, host, port, timeout=5.0):
        super().__init__(state_action_encoder)
        channel = grpc.insecure_channel('{}:{}'.format(host, port))
        self.prediction_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        self.timeout = timeout

        self.representation_io = self.get_io('representation')
        self.representation = lambda observation: self.gRPC('representation', self.representation_io, observation)[0]

        self.dynamics_io = self.get_io('dynamics')
        self.dynamics = lambda encoded_state_action: self.gRPC('dynamics', self.dynamics_io, encoded_state_action)

        self.prediction_io = self.get_io('prediction')
        self.prediction = lambda hidden_state_batch: self.gRPC('prediction', self.prediction_io, hidden_state_batch)

    def get_io(self, sub_network):
        metadata_request = get_model_metadata_pb2.GetModelMetadataRequest()
        metadata_request.model_spec.name = sub_network
        metadata_request.metadata_field.append("signature_def")
        result = self.prediction_service.GetModelMetadata(metadata_request, self.timeout)

        signature_def_map = get_model_metadata_pb2.SignatureDefMap()
        result.metadata['signature_def'].Unpack(signature_def_map)
        default_signature_def = signature_def_map.signature_def['serving_default']
        return list(default_signature_def.inputs.keys()), list(default_signature_def.outputs.keys())

    def gRPC(self, sub_network, inputs_outputs, tensor):
        inputs, outputs = inputs_outputs
        request = predict_pb2.PredictRequest()
        request.model_spec.name = sub_network
        request.inputs[inputs[0]].CopyFrom(tf.make_tensor_proto(tensor))

        response = self.prediction_service.Predict(request, self.timeout)

        return [tf.make_ndarray(response.outputs[output]) for output in outputs]
