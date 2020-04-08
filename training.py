from storage_replay import *
from evaluation import *
import tensorflow as tf
import os


def tensorboard_summary(model, line_length=100):
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


def evaluation_logger(summary_writer, step, evaluation_stats):
    if summary_writer:
        with summary_writer.as_default():
            for player, result in evaluation_stats.items():
                tf.summary.scalar(name='Evaluation/{}'.format(player), data=result, step=step)


def self_play_logger(summary_writer, step, num_games, num_positions, num_unique):
    if summary_writer:
        with summary_writer.as_default():
            tf.summary.scalar(name='Self-play/Number of games', data=num_games, step=step)
            tf.summary.scalar(name='Self-play/Number of positions', data=num_positions, step=step)
            tf.summary.scalar(name='Self-play/Number of unique games', data=num_unique, step=step)


def tensorboard_logger(config, checkpoint_path, network):
    if checkpoint_path:
        checkpoint_dir = os.path.join(checkpoint_path, config.name, timestamp())
        log_dir = os.path.join(checkpoint_dir, 'logs')

    #     # Graphs for all networks
    #     network = storage.latest_network()
    #     # tensorboard_graph(network.representation, log_dir)
    #     # tensorboard_graph(network.dynamics, log_dir)
    #     # tensorboard_graph(network.prediction, log_dir)
    #
        # Network and game information
        summary_writer = tf.summary.create_file_writer(log_dir)
        with summary_writer.as_default():
            tf.summary.text(name='Configuration', data='| Key | Value |\n|-----|-------|\n' + '\n'.join(
                '| {} | {} |'.format(key, value) for key, value in config.__dict__.items() if
                key not in ['reward_loss', 'value_loss', 'visit_softmax_temperature_fn', 'action_space', 'game_class', 'network_class', 'game_params']), step=0)
            tf.summary.text(name='Game parameters', data='| Key | Value |\n|-----|-------|\n' + '\n'.join(
                '| {} | {} |'.format(key, value) for key, value in config.game_params.items() if
                key not in []), step=0)
            tf.summary.text(name='Networks/Representation', data=tensorboard_summary(network.representation), step=0)
            tf.summary.text(name='Networks/Dynamics', data=tensorboard_summary(network.dynamics), step=0)
            tf.summary.text(name='Networks/Prediction', data=tensorboard_summary(network.prediction), step=0)
    else:
        checkpoint_dir = None
        summary_writer = None
    return checkpoint_dir, summary_writer


def synchronous_train_network(config, network, num_games, num_steps, num_eval_games, checkpoint_path=None):
    replay_buffer = ReplayBuffer(config)
    optimizer = tf.keras.optimizers.Adam(lr=config.learning_rate)

    # Tensorboard logging
    checkpoint_dir, summary_writer = tensorboard_logger(config, checkpoint_path, network)

    # Actual training
    for step in range(config.training_steps):
        if step % num_steps == 0:
            batch_selfplay(config, replay_buffer, network, num_games)
            self_play_logger(summary_writer, step=step,
                             num_games=len(replay_buffer.buffer),
                             num_positions=replay_buffer.num_positions,
                             num_unique=replay_buffer.num_unique())

        if step % config.checkpoint_interval == 0:
            if checkpoint_dir:
                network.save_weights(os.path.join(checkpoint_dir, '{}_it{}'.format(config.name, step)))

            if num_eval_games:
                evaluation_stats = evaluate_agent(config, network, num_eval_games)
                evaluation_logger(summary_writer, step=step, evaluation_stats=evaluation_stats)

            # learning_rate = config.lr_init*config.lr_decay_rate ** (network.training_steps() / config.lr_decay_steps)
            #      optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=config.momentum)

            # Learning-rate logging
            # if summary_writer:
            #     with summary_writer.as_default():
            #         tf.summary.scalar(name='Learning rate', data=learning_rate, step=i)

        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps, config.discount)
        metrics = batch_update_weights(config, network, optimizer, batch)
        loss_logger(summary_writer, step=step, metrics=metrics)

    if checkpoint_dir:
        network.save_weights(os.path.join(checkpoint_dir, '{}_it{}'.format(config.name, config.training_steps)))


def train_network(config, server_address, num_eval_games, tensorboard_logpath=None):
    client = MuZeroClient(config, server_address)
    network = client.latest_network()
    optimizer = tf.keras.optimizers.Adam(lr=config.learning_rate)

    # Tensorboard logging
    tensorboard_logdir, summary_writer = tensorboard_logger(config, tensorboard_logpath, network)
    print('Tensorboard logging in directory {}'.format(tensorboard_logdir))

    for step in range(config.training_steps):
        if step % config.checkpoint_interval == 0:
            client.save_network(network)

            if num_eval_games:
                evaluation_stats = evaluate_agent(config, network, num_eval_games)
                evaluation_logger(summary_writer, step=step, evaluation_stats=evaluation_stats)

        server_information = client.information()
        self_play_logger(summary_writer, step=step,
                         num_games=server_information['num_games'],
                         num_positions=server_information['num_positions'],
                         num_unique=server_information['num_unique'])

        batch = client.sample_batch()
        metrics = batch_update_weights(config, network, optimizer, batch)
        loss_logger(summary_writer, step=step, metrics=metrics)
    client.save_network(network)


def scale_gradient(tensor, scale):
    """
    Scales the gradient for the backward pass.
    """
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def batch_update_weights(config, network, optimizer, batch):
    value_loss_metric = tf.keras.metrics.Mean('Losses/Value', dtype=tf.float32)
    reward_loss_metric = tf.keras.metrics.Mean('Losses/Reward', dtype=tf.float32)
    policy_loss_metric = tf.keras.metrics.Mean('Losses/Policy', dtype=tf.float32)
    toplay_loss_metric = tf.keras.metrics.Mean('Losses/ToPlay', dtype=tf.float32)
    regularization_metric = tf.keras.metrics.Sum('Losses/Regularization', dtype=tf.float32)

    with tf.GradientTape() as tape:
        batch_images, batch_actions, batch_targets = map(np.array, zip(*batch))
        batch_targets = np.transpose(batch_targets, axes=(1, 0, 2))

        batch_initial_inference = network.initial_inference(batch_images)

        batch_predictions = [(1.,
                              batch_initial_inference.value,
                              None,
                              batch_initial_inference.policy_logits,
                              None,
                              False
                              )]
        batch_hidden_state = batch_initial_inference.hidden_state

        for batch_action in batch_actions.T:
            batch_recurrent_inference = network.recurrent_inference(batch_hidden_state, batch_action)

            batch_predictions.append((1./config.num_unroll_steps,
                                      batch_recurrent_inference.value,
                                      batch_recurrent_inference.reward,
                                      batch_recurrent_inference.policy_logits,
                                      batch_recurrent_inference.to_play,
                                      True
                                      ))
            batch_hidden_state = scale_gradient(batch_recurrent_inference.hidden_state, 0.5)

        loss = tf.constant(0, dtype=np.float32)
        for predictions, targets in zip(batch_predictions, batch_targets):
            gradient_scale, batch_value, batch_reward, batch_policy_logits, batch_toplay, predict_reward = predictions
            batch_target_value, batch_target_reward, batch_target_policy_logits, batch_target_toplay = zip(*targets)

            value_loss = config.value_loss(batch_target_value, batch_value)
            value_loss_metric(value_loss)

            policy_loss = tf.keras.losses.categorical_crossentropy(batch_target_policy_logits, batch_policy_logits, from_logits=True)
            policy_loss_metric(policy_loss)

            loss += tf.math.reduce_mean(scale_gradient(value_loss + policy_loss, gradient_scale))/(config.num_unroll_steps+1)

            if predict_reward:
                reward_loss = config.reward_loss(batch_target_reward, batch_reward)
                reward_loss_metric(reward_loss)

                toplay_loss = tf.keras.losses.sparse_categorical_crossentropy(batch_target_toplay, batch_toplay, from_logits=True)
                toplay_loss_metric(toplay_loss)
                loss += tf.math.reduce_mean(scale_gradient(reward_loss + toplay_loss, gradient_scale))/config.num_unroll_steps

        for weights in network.get_weights():
            l2_regularization = config.weight_decay * tf.nn.l2_loss(weights)
            regularization_metric(l2_regularization)
            loss += l2_regularization

    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))
    network.steps += 1

    return value_loss_metric, reward_loss_metric, policy_loss_metric, toplay_loss_metric, regularization_metric
