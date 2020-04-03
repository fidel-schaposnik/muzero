import tensorflow as tf
from environment import *


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
    x = tf.keras.layers.BatchNormalization(name=name + '_norm')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_relu')(x)
    x = tf.keras.layers.Flatten(name=name + '_flatten')(x)
    x = tf.keras.layers.Dense(num_logits, name=name + '_dense', activation='relu')(x)
    return x


def scalar_head(name, inputs, num_outputs, num_filters, kernel_size, hidden_size):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size, padding='same', name=name + '_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(name=name + '_norm')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_relu')(x)
    x = tf.keras.layers.Flatten(name=name + '_flatten')(x)
    x = tf.keras.layers.Dense(hidden_size, activation='relu', name=name + '_hidden')(x)
    x = tf.keras.layers.Dense(num_outputs, activation='relu', name=name + '_output')(x)
    return x


def hidden_state_normalization(tensor):
    a = tf.math.reduce_min(tensor, axis=(1, 2, 3), keepdims=True)
    b = tf.math.reduce_max(tensor, axis=(1, 2, 3), keepdims=True)
    return (tensor - a) / tf.where(a == b, 1., b - a)


def prediction_network(name, input_shape, num_logits, num_players,
                       tower_height=19, conv_filters=256, conv_kernel_size=(3, 3),
                       policy_filters=2, policy_kernel_size=(1, 1),
                       value_filters=1, value_kernel_size=(1, 1), value_hidden_size=256):
    inputs = tf.keras.Input(shape=input_shape, name=name + '_input_hidden_state')
    residual_output = residual_tower(name=name + '_tower', inputs=inputs,
                                     height=tower_height, num_filters=conv_filters, kernel_size=conv_kernel_size)
    policy_output = logits_head(name=name + '_policy', inputs=residual_output, num_logits=num_logits,
                                num_filters=policy_filters, kernel_size=policy_kernel_size)
    value_output = scalar_head(name=name + '_value', inputs=residual_output, num_outputs=num_players,
                               num_filters=value_filters, kernel_size=value_kernel_size, hidden_size=value_hidden_size)
    return tf.keras.Model(inputs=inputs, outputs=[policy_output, value_output], name=name)


def dynamics_network(name, input_shape, num_players,
                     tower_height=19, conv_filters=256, conv_kernel_size=(3, 3),
                     reward_filters=1, reward_kernel_size=(1, 1), reward_hidden_size=256,
                     toplay_filters=1, toplay_kernel_size=(1, 1)
                     ):

    inputs = tf.keras.Input(shape=input_shape, name=name + '_input_hidden_state')

    residual_output = residual_tower(name=name + '_tower', inputs=inputs,
                                     height=tower_height, num_filters=conv_filters, kernel_size=conv_kernel_size)
    hidden_state = tf.keras.layers.Lambda(hidden_state_normalization, name=name + '_hidden_state_norm')(residual_output)

    reward_output = scalar_head(name=name + '_reward', inputs=hidden_state, num_outputs=num_players,
                                num_filters=reward_filters, kernel_size=reward_kernel_size, hidden_size=reward_hidden_size)

    toplay_logits = logits_head(name=name + '_toplay', inputs=hidden_state, num_logits=num_players,
                                num_filters=toplay_filters, kernel_size=toplay_kernel_size)

    return tf.keras.Model(inputs=inputs, outputs=[hidden_state, reward_output, toplay_logits], name=name)


def representation_network(name, input_shape, conv_filters=256, conv_kernel_size=(3, 3), tower_height=19):
    inputs = tf.keras.Input(shape=input_shape, name=name + '_input_image')
    residual_output = residual_tower(name=name + '_tower', inputs=inputs,
                                     height=tower_height, num_filters=conv_filters, kernel_size=conv_kernel_size)
    hidden_state = tf.keras.layers.Lambda(hidden_state_normalization, name=name + '_hidden_state_norm')(residual_output)
    return tf.keras.Model(inputs=inputs, outputs=hidden_state, name=name)


class NetworkOutput(object):
    def __init__(self, value, reward, policy_logits, hidden_state, to_play):
        self.value = value
        self.reward = reward
        self.policy_logits = policy_logits
        self.hidden_state = hidden_state
        self.to_play = to_play

    def split_batch(self):
        return [NetworkOutput(
                              {Player(i): value for i, value in enumerate(values)},
                              {Player(i): reward for i, reward in enumerate(rewards)},
                              {Action(i): logit for i, logit in enumerate(policy_logits)},
                              hidden_state,
                              to_play
                              )
                for values, rewards, policy_logits, hidden_state, to_play in
                zip(self.value, self.reward, self.policy_logits, self.hidden_state, self.to_play)
                ]


class Network:
    """
    A class for grouping all of MuZero neural networks.
    """

    def __init__(self):
        self.steps = 0
        self.representation = None
        self.dynamics = None
        self.prediction = None

    def training_steps(self):
        """
        Returns the number of steps this network has been trained for.
        """
        return self.steps

    def get_weights(self):
        """
        Returns the weights of this network.
        """
        weights = []
        for sub_network in [self.representation, self.dynamics, self.prediction]:
            weights.extend(sub_network.trainable_weights)
        return weights

    def save_weights(self, filepath_prefix):
        self.representation.save_weights(filepath_prefix + '_rep.h5')
        self.dynamics.save_weights(filepath_prefix + '_dyn.h5')
        self.prediction.save_weights(filepath_prefix + '_pre.h5')

    def load_weights(self, filepath_prefix):
        self.representation.load_weights(filepath_prefix + '_rep.h5')
        self.dynamics.load_weights(filepath_prefix + '_dyn.h5')
        self.prediction.load_weights(filepath_prefix + '_pre.h5')

    def summary(self):
        self.representation.summary()
        self.dynamics.summary()
        self.prediction.summary()

    def hidden_state_shape(self, batch_size=None):
        raise ImplementationError('hidden_state_shape', 'Network')

    def initial_inference(self, batch_image):
        raise ImplementationError('initial_inference', 'Network')

    def recurrent_inference(self, batch_hidden_state, batch_action):
        raise ImplementationError('recurrent_inference', 'Network')

    def __str__(self):
        return 'Network({}, trining_steps={})'.format(
            {model.name: {'parameters': model.count_params(),
                          'input_shape': model.input_shape,
                          'output_shape': model.output_shape} for model in [self.representation, self.dynamics, self.prediction]},
            self.training_steps())