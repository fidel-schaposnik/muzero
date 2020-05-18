import tensorflow as tf
import numpy as np
import os
import argparse
import time

from utils import timestamp, load_game
from replay_buffer import RemoteReplayBuffer


def tensorboard_model_summary(model, line_length=100):
    lines = []
    model.summary(print_fn=lambda line: lines.append(line), line_length=line_length)
    lines.insert(3, '-'*line_length)
    positions = [lines[2].find(col) for col in ['Layer', 'Output', 'Param', 'Connected']]
    positions.append(line_length)
    table = ['|'+'|'.join([line[positions[i]:positions[i+1]] for i in range(len(positions)-1)])+'|' for line in lines[2:-4] if line[0] not in ['=', '_']]
    result = '# Model summary\n' + '\n'.join(table) + '\n\n# Parameter summary\n' + '\n\n'.join(lines[-4:-1])
    return result


def loss_logger(summary_writer, step, metrics):
    print(('{},' + ','.join('{:.4f}' for _ in range(len(metrics) + 1))).format(step, *[metric.result() for metric in metrics], sum(metric.result() for metric in metrics)))
    if summary_writer:
        with summary_writer.as_default():
            tf.summary.scalar(name='Losses/Total', data=sum(metric.result() for metric in metrics), step=step)
            for metric in metrics:
                tf.summary.scalar(name=metric.name, data=metric.result(), step=step)


# def evaluation_logger(summary_writer, step, evaluation_stats):
#     if summary_writer:
#         with summary_writer.as_default():
#             for player, result in evaluation_stats.items():
#                 tf.summary.scalar(name='Evaluation/{}'.format(player), data=result, step=step)


# def self_play_logger(summary_writer, step, num_games, num_positions, num_unique):
#     if summary_writer:
#         with summary_writer.as_default():
#             tf.summary.scalar(name='Self-play/Number of games', data=num_games, step=step)
#             tf.summary.scalar(name='Self-play/Number of positions', data=num_positions, step=step)
#             tf.summary.scalar(name='Self-play/Number of unique games', data=num_unique, step=step)


def tensorboard_logger(training_config, checkpoint_path, network):
    if checkpoint_path:
        checkpoint_dir = os.path.join(checkpoint_path, training_config.game_config.name, timestamp())
        checkpoint_prefix = os.path.join(checkpoint_dir, '{}_ckpt'.format(training_config.game_config.name))
        log_dir = os.path.join(checkpoint_dir, 'logs')

        # Network and game information
        summary_writer = tf.summary.create_file_writer(log_dir)
        with summary_writer.as_default():
            tf.summary.text(name='Training configuration',
                            data='| Key | Value |\n|-----|-------|\n' + '\n'.join(
                '| {} | {} |'.format(key, value) for key, value in training_config.__dict__.items() if key not in
                ['game_config', 'value_loss', 'reward_loss', 'optimizer']), step=0)
            tf.summary.text(name='Game configuration',
                            data='| Key | Value |\n|-----|-------|\n' + '\n'.join(
                '| {} | {} |'.format(key, value) for key, value in training_config.game_config.__dict__.items()), step=0)
            tf.summary.text(name='Networks/Representation', data=tensorboard_model_summary(network.representation), step=0)
            tf.summary.text(name='Networks/Dynamics', data=tensorboard_model_summary(network.dynamics), step=0)
            tf.summary.text(name='Networks/Prediction', data=tensorboard_model_summary(network.prediction), step=0)
    else:
        checkpoint_prefix = None
        summary_writer = None
    return checkpoint_prefix, summary_writer


def train_network(training_config, network, replay_buffer, saved_models_path=None, logging_path=None):
    optimizer = training_config.optimizer

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     representation=network.representation,
                                     dynamics=network.dynamics,
                                     prediction=network.prediction)

    # Tensorboard logging
    checkpoint_prefix, summary_writer = tensorboard_logger(training_config, logging_path, network)

    for step in range(training_config.training_steps):
        batch = replay_buffer.sample_batch(batch_size=training_config.batch_size,
                                           num_unroll_steps=training_config.num_unroll_steps,
                                           td_steps=training_config.td_steps,
                                           discount=training_config.game_config.discount)
        metrics = batch_update_weights(training_config, network, optimizer, batch)
        loss_logger(summary_writer, step=step, metrics=metrics)

        if network.training_steps() % training_config.checkpoint_interval == 0:
            if saved_models_path:
                network.save_model(saved_models_path)
            if checkpoint_prefix:
                checkpoint.save(checkpoint_prefix)


def scale_gradient(tensor, scale):
    """
    Scales the gradient for the backward pass.
    """
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def batch_update_weights(training_config, network, optimizer, batch):
    value_loss_metric = tf.keras.metrics.Mean('Losses/Value', dtype=tf.float32)
    reward_loss_metric = tf.keras.metrics.Mean('Losses/Reward', dtype=tf.float32)
    policy_loss_metric = tf.keras.metrics.Mean('Losses/Policy', dtype=tf.float32)
    regularization_metric = tf.keras.metrics.Sum('Losses/Regularization', dtype=tf.float32)

    with tf.GradientTape() as tape:
        batch_images, batch_actions, batch_targets = map(np.array, zip(*batch))
        batch_targets = np.transpose(batch_targets, axes=(1, 0, 2))

        batch_initial_inference = network.initial_inference(batch_images, training=True)

        batch_predictions = [(1.,
                              batch_initial_inference.batch_value,
                              None,
                              batch_initial_inference.batch_policy_logits,
                              False
                              )]
        batch_hidden_state = batch_initial_inference.batch_hidden_state

        for batch_action in batch_actions.T:
            batch_recurrent_inference = network.recurrent_inference(batch_hidden_state, batch_action, training=True)
            batch_predictions.append((1./training_config.num_unroll_steps,
                                      batch_recurrent_inference.batch_value,
                                      batch_recurrent_inference.batch_reward,
                                      batch_recurrent_inference.batch_policy_logits,
                                      True
                                      ))
            batch_hidden_state = scale_gradient(batch_recurrent_inference.batch_hidden_state, 0.5)

        loss = tf.constant(0, dtype=tf.float32)
        for predictions, targets in zip(batch_predictions, batch_targets):
            gradient_scale, batch_value, batch_reward, batch_policy_logits, predict_reward = predictions
            batch_target_value, batch_target_reward, batch_target_policy_logits = zip(*targets)

            value_loss = training_config.value_loss_decay * training_config.value_loss(batch_target_value, batch_value)
            value_loss_metric(value_loss)

            policy_loss = tf.keras.losses.categorical_crossentropy(batch_target_policy_logits, batch_policy_logits, from_logits=True)
            policy_loss_metric(policy_loss)

            loss += tf.math.reduce_mean(scale_gradient(value_loss + policy_loss, gradient_scale))/(training_config.num_unroll_steps+1)

            if predict_reward:
                reward_loss = training_config.reward_loss_decay * training_config.reward_loss(batch_target_reward, batch_reward)
                reward_loss_metric(reward_loss)

                loss += tf.math.reduce_mean(scale_gradient(reward_loss, gradient_scale))/training_config.num_unroll_steps

        l2_regularization = training_config.regularization_decay * sum(tf.nn.l2_loss(weights) for weights in network.get_weights())
        regularization_metric(l2_regularization)
        loss += l2_regularization

    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))
    network.steps += 1

    return value_loss_metric, reward_loss_metric, policy_loss_metric, regularization_metric


if __name__ == '__main__':
    from importlib import import_module
    game_module = import_module('games.cartpole')

    parser = argparse.ArgumentParser(description='MuZero Training Client')
    parser.add_argument('--game', type=str, required=True,
                        help='One of the games implemented in the games/ directory')
    parser.add_argument('--replay_buffer', type=str, required=True,
                        help='IP:Port for gRPC communication with a replay buffer server')
    parser.add_argument('--min_games', type=int, default=1,
                        help='Minimum number of games required to start training')
    parser.add_argument('--saved_models_path', type=str, required=True,
                        help='Path to the models/ directory served by tensorflow serving.')
    parser.add_argument('--logging_path', type=str, default=None,
                        help='Path to the directory where checkpoints and logs shall be stored.')
    args = parser.parse_args()

    if not os.path.isdir(args.logging_path):
        parser.error('{} is not a valid directory!'.format(args.logging_path))
    else:
        config = load_game(args.game, parser)
        local_network = config.network_config.make_uniform_network()
        remote_replay_buffer = RemoteReplayBuffer(ip_port=args.replay_buffer)

        local_network.save_model(args.saved_models_path)

        while remote_replay_buffer.stats()['num_games'] < args.min_games:
            print('Waiting for {} games to be available on the replay buffer...'.format(args.min_games))
            time.sleep(60)

        train_network(training_config=config.training_config,
                      network=local_network,
                      replay_buffer=remote_replay_buffer,
                      saved_models_path=args.saved_models_path,
                      logging_path=args.logging_path)
