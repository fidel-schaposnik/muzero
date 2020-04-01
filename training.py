from storage_replay import *
from evaluation import *
import tensorflow as tf
import os, datetime


def timestamp():
    return datetime.datetime.now().strftime("%d-%m-%Y--%H-%M")


def tensorboard_summary(model, line_length=100):
    lines = []
    model.summary(print_fn=lambda line: lines.append(line), line_length=line_length)
    lines.insert(3, '-'*line_length)
    positions = [lines[2].find(col) for col in ['Layer', 'Output', 'Param', 'Connected']]
    positions.append(line_length)
    table = ['|'+'|'.join([line[positions[i]:positions[i+1]] for i in range(len(positions)-1)])+'|' for line in lines[2:-4] if line[0] not in ['=', '_']]
    result = '# Model summary\n' + '\n'.join(table) + '\n\n# Parameter summary\n' + '\n\n'.join(lines[-4:-1])
    return result


def synchroneous_train_network(config, storage, num_games, num_steps, num_eval_games, checkpoint_path=None):
    replay_buffer = ReplayBuffer(config)
    network = storage.latest_network()
    optimizer = tf.keras.optimizers.Adam(lr=config.learning_rate)

    # Tensorboard logging
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

    # Actual training
    for step in range(config.training_steps):
        if step % num_steps == 0:
            storage.save_network(step, network)

            batch_selfplay(config, replay_buffer, network, num_games)

            # Self-play logging
            if summary_writer:
                with summary_writer.as_default():
                    tf.summary.scalar(name='Self-play/Number of games', data=len(replay_buffer.buffer), step=step)
                    tf.summary.scalar(name='Self-play/Number of positions', data=replay_buffer.num_positions, step=step)

        if step % config.checkpoint_interval == 0:
            if checkpoint_dir:
                network.save_weights(os.path.join(checkpoint_dir, '{}_it{}'.format(config.name, step)))

            if num_eval_games:
                evaluation_stats = evaluate_agents(config, ['random', network], num_eval_games)

                # Evaluation logging
                if summary_writer:
                    with summary_writer.as_default():
                        tf.summary.text(name='Evaluation',
                                        data='{:.2f}% won games, {:.2f}% lost games, {:.2f}% drawn games'.format(
                                            100 * evaluation_stats[-1] / num_eval_games,
                                            100 * evaluation_stats[1] / num_eval_games,
                                            100 * evaluation_stats[0] / num_eval_games),
                                        step=step)

            # learning_rate = config.lr_init*config.lr_decay_rate ** (network.training_steps() / config.lr_decay_steps)
            #      optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=config.momentum)

            # Learning-rate logging
            # if summary_writer:
            #     with summary_writer.as_default():
            #         tf.summary.scalar(name='Learning rate', data=learning_rate, step=i)

        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps, config.discount)
        metrics = batch_update_weights(config, network, optimizer, batch)

        print(('{},'+','.join('{:.4f}' for _ in range(len(metrics)+1))).format(step, *[metric.result() for metric in metrics], sum(metric.result() for metric in metrics)))
        if summary_writer:
            with summary_writer.as_default():
                tf.summary.scalar(name='Losses/Total', data=sum(metric.result() for metric in metrics), step=step)
                for metric in metrics:
                    tf.summary.scalar(name=metric.name, data=metric.result(), step=step)

    storage.save_network(config.training_steps, network)
    if checkpoint_dir:
        network.save_weights(os.path.join(checkpoint_dir, '{}_it{}'.format(config.name, config.training_steps)))


def scale_gradient(tensor, scale):
    """Scales the gradient for the backward pass."""
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

            loss += tf.math.reduce_mean(value_loss + policy_loss)/(config.num_unroll_steps+1)

            if predict_reward:
                reward_loss = config.reward_loss(batch_target_reward, batch_reward)
                reward_loss_metric(reward_loss)

                toplay_loss = tf.keras.losses.sparse_categorical_crossentropy(batch_target_toplay, batch_toplay, from_logits=True)
                toplay_loss_metric(toplay_loss)
                loss += tf.math.reduce_mean(reward_loss + toplay_loss)/config.num_unroll_steps

            # loss += scale_gradient(value_loss + reward_loss + tf.math.reduce_mean(policy_loss), gradient_scale)

        for weights in network.get_weights():
            l2_regularization = config.weight_decay * tf.nn.l2_loss(weights)
            regularization_metric(l2_regularization)
            loss += l2_regularization

    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))
    network.steps += 1

    # Logging
    # if summary_writer:
    #     with summary_writer.as_default():
    #         tf.summary.scalar(name='Losses/Total', data=loss.numpy(), step=network.training_steps())
    #         tf.summary.scalar(name='Losses/Value', data=value_loss_metric.result(), step=network.training_steps())
    #         tf.summary.scalar(name='Losses/Reward', data=reward_loss_metric.result(), step=network.training_steps())
    #         tf.summary.scalar(name='Losses/Policy', data=policy_loss_metric.result(), step=network.training_steps())
    #         tf.summary.scalar(name='Losses/Regularization', data=regularization_loss.numpy(), step=network.training_steps())

    # print('{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'.format(network.training_steps(), value_loss_metric.result(),
    #                                                      reward_loss_metric.result(), policy_loss_metric.result(),
    #                                                      toplay_loss_metric.result(),
    #                                                      regularization_metric.result(), loss.numpy()))
    return value_loss_metric, reward_loss_metric, policy_loss_metric, toplay_loss_metric, regularization_metric
