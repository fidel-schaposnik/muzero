import re
import tensorflow as tf

# For type annotations
from typing import List, Dict, Optional, Union

from muzero.config import MuZeroConfig
from muzero.network import Network
from muzero.replay_buffer import ReplayBuffer
from muzero.replay_buffer_services import RemoteReplayBuffer


def scale_gradient(tensor: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    """
    Scales the gradient for the backward pass.
    """
    return tensor * scale + tf.stop_gradient(tensor) * (tf.ones_like(scale) - scale)


def build_unrolled_model(config: MuZeroConfig, network: Network) -> tf.keras.Model:
    gradient_scale = tf.constant(1 / config.training_config.num_unroll_steps)

    observation = tf.keras.Input(shape=network.state_preprocessing.input_shape[1:],
                                 name=config.network_config.OBSERVATION)
    unroll_actions = tf.keras.Input(shape=(config.training_config.num_unroll_steps, ),
                                    name=config.network_config.UNROLL_ACTIONS)
    hidden_state, value, policy_logits = network.initial_inference_model(observation)

    unrolled_rewards = []
    unrolled_values = [value]
    unrolled_policy_logits = [policy_logits]
    for action in tf.transpose(unroll_actions):
        hidden_state = scale_gradient(hidden_state, tf.constant(0.5))
        hidden_state, reward, value, policy_logits = network.recurrent_inference_model([hidden_state, action])

        unrolled_rewards.append(scale_gradient(reward, gradient_scale))
        unrolled_values.append(scale_gradient(value, gradient_scale))
        unrolled_policy_logits.append(scale_gradient(policy_logits, gradient_scale))

    unrolled_rewards = tf.keras.layers.Lambda(
        lambda inputs: tf.stack(inputs, axis=1),
        name=config.network_config.UNROLLED_REWARDS)(unrolled_rewards)
    unrolled_values = tf.keras.layers.Lambda(
        lambda inputs: tf.stack(inputs, axis=1),
        name=config.network_config.UNROLLED_VALUES)(unrolled_values)
    unrolled_policy_logits = tf.keras.layers.Lambda(
        lambda inputs: tf.stack(inputs, axis=1),
        name=config.network_config.UNROLLED_POLICY_LOGITS)(unrolled_policy_logits)
    return tf.keras.Model(inputs=[observation, unroll_actions],
                          outputs=[unrolled_rewards, unrolled_values, unrolled_policy_logits],
                          name=config.network_config.UNROLLED_MODEL)


class MuZeroCallback(tf.keras.callbacks.Callback):
    def __init__(self, network: Network, saved_models_path: str,
                 checkpoint_manager: Optional[tf.train.CheckpointManager]) -> None:
        super().__init__()
        self.network: Network = network
        self.saved_models_path: str = saved_models_path
        self.checkpoint_manager: Optional[tf.train.CheckpointManager] = checkpoint_manager

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None) -> None:
        self.network.save_tfx_models(self.saved_models_path)
        print(
            f'Saved network with {self.network.training_steps()} steps to {self.saved_models_path}')
        if self.checkpoint_manager:
            self.checkpoint_manager.save()
            print(f'Saved checkpoint to {self.checkpoint_manager.latest_checkpoint}')

    def on_train_batch_end(self, batch: int, logs: Dict[str, float] = None) -> None:
        self.network.steps.assign_add(1)


class LossLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, config: MuZeroConfig, network: Network,
                 writer: tf.summary.SummaryWriter) -> None:
        super().__init__()
        self.config: MuZeroConfig = config
        self.network: Network = network
        self.writer: tf.summary.SummaryWriter = writer

        self.value_loss_decay = self.config.value_config.loss_decay
        self.reward_loss_decay = self.config.reward_config.loss_decay

    def on_train_batch_end(self, batch: int, logs: Dict[str, float] = None) -> None:
        total_loss = logs['loss']
        reward_loss = self.reward_loss_decay * logs[f'{self.config.network_config.UNROLLED_REWARDS}_loss']
        value_loss = self.value_loss_decay * logs[f'{self.config.network_config.UNROLLED_VALUES}_loss']
        policy_loss = logs[f'{self.config.network_config.UNROLLED_POLICY_LOGITS}_loss']
        regularization = total_loss - reward_loss - value_loss - policy_loss
        with self.writer.as_default():
            tf.summary.scalar(name='Losses/Total',
                              data=total_loss,
                              step=self.network.training_steps())
            tf.summary.scalar(name='Losses/Reward',
                              data=reward_loss,
                              step=self.network.training_steps())
            tf.summary.scalar(name='Losses/Value',
                              data=value_loss,
                              step=self.network.training_steps())
            tf.summary.scalar(name='Losses/Policy',
                              data=policy_loss,
                              step=self.network.training_steps())
            tf.summary.scalar(name='Losses/Regularization',
                              data=regularization,
                              step=self.network.training_steps())


class ReplayBufferLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, network: Network, replay_buffer: Union[ReplayBuffer, RemoteReplayBuffer],
                 replay_buffer_loginterval: int, writer: tf.summary.SummaryWriter) -> None:
        super().__init__()
        self.network: Network = network
        self.replay_buffer: Union[ReplayBuffer, RemoteReplayBuffer] = replay_buffer
        self.replay_buffer_loginterval: int = replay_buffer_loginterval
        self.writer: tf.summary.SummaryWriter = writer
        self.network_to_log: Optional[int] = None

    def on_train_batch_end(self, batch: int, logs: Dict[str, float] = None):
        if self.network.training_steps() % self.replay_buffer_loginterval == 0:
            detailed_stats = self.replay_buffer.detailed_stats()
            with self.writer.as_default():
                agents = set()
                for field, value in detailed_stats.items():
                    regex = re.search('Agents/([^:]*)', field)
                    if regex is not None:
                        agents.add(regex.group(1))
                    elif 'Networks' not in field:
                        tf.summary.scalar(field, data=value, step=self.network.training_steps())

                for agent_id in agents:
                    games_played = int(detailed_stats[f'Agents/{agent_id}: games played'])
                    average_total_value = detailed_stats[f'Agents/{agent_id}: average total value']
                    tf.summary.scalar(f'Agents/{agent_id}: average total value',
                                      data=average_total_value,
                                      step=games_played)

                if self.network_to_log is not None:
                    if f'Networks/{self.network_to_log}: games played' in detailed_stats.keys():
                        games_played = detailed_stats[
                            f'Networks/{self.network_to_log}: games played']
                        tf.summary.scalar(name='Networks/Games played',
                                          data=games_played,
                                          step=self.network_to_log)
                    if f'Networks/{self.network_to_log}: average total value' in detailed_stats.keys():
                        average_total_value = detailed_stats[
                            f'Networks/{self.network_to_log}: average total value']
                        tf.summary.scalar(name='Networks/Average total value',
                                          data=average_total_value,
                                          step=self.network_to_log)
                    self.network_to_log = None

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None):
        self.network_to_log = self.network.training_steps() - self.params['steps']


def train_network(
        config: MuZeroConfig,
        network: Network,
        optimizer: tf.keras.optimizers.Optimizer,
        replay_buffer: Union[ReplayBuffer, RemoteReplayBuffer],
        saved_models_path: str,
        writer: Optional[tf.summary.SummaryWriter] = None,
        checkpoint_manager: Optional[tf.train.CheckpointManager] = None) -> Dict[str, List[float]]:

    replay_buffer_loginterval = config.training_config.replay_buffer_loginterval
    unrolled_model = build_unrolled_model(config, network)
    unrolled_model.compile(
        loss={
            config.network_config.UNROLLED_REWARDS: config.reward_config.loss,
            config.network_config.UNROLLED_VALUES: config.value_config.loss,
            config.network_config.UNROLLED_POLICY_LOGITS: tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        },
        loss_weights={
            config.network_config.UNROLLED_REWARDS: config.reward_config.loss_decay,
            config.network_config.UNROLLED_VALUES: config.value_config.loss_decay,
            config.network_config.UNROLLED_POLICY_LOGITS: 1.0
        },
        optimizer=optimizer,
        steps_per_execution=config.training_config.steps_per_execution)

    dataset = replay_buffer.as_dataset(batch_size=config.training_config.batch_size)

    muzero_callback = MuZeroCallback(network=network,
                                     saved_models_path=saved_models_path,
                                     checkpoint_manager=checkpoint_manager)
    callbacks = [muzero_callback]
    if writer:
        loss_logger = LossLoggerCallback(config=config, network=network, writer=writer)
        callbacks.append(loss_logger)
        if replay_buffer_loginterval is not None:
            replay_buffer_callback = ReplayBufferLoggerCallback(
                network=network,
                replay_buffer=replay_buffer,
                replay_buffer_loginterval=replay_buffer_loginterval,
                writer=writer)
            callbacks.append(replay_buffer_callback)

    num_epochs = config.training_config.training_steps // config.training_config.checkpoint_interval
    history = unrolled_model.fit(dataset,
                                 epochs=num_epochs,
                                 steps_per_epoch=config.training_config.checkpoint_interval,
                                 callbacks=callbacks)
    return history.history
