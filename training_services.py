import os
import time
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from grpc import RpcError

from muzero.utils import CommandLineParser
from muzero.replay_buffer_services import RemoteReplayBuffer
from muzero.training import train_network


def tensorboard_model_summary(model: tf.keras.Model, line_length: int = 100) -> str:
    lines = []
    model.summary(print_fn=lambda line: lines.append(line), line_length=line_length)
    lines.insert(3, '-'*line_length)
    positions = [lines[2].find(col) for col in ['Layer', 'Output', 'Param', 'Connected']]
    positions.append(line_length)
    table = ['|'+'|'.join([line[positions[i]:positions[i+1]] for i in range(len(positions)-1)])+'|' for line in lines[2:-4] if line[0] not in ['=', '_']]
    result = '# Model summary\n' + '\n'.join(table) + '\n\n# Parameter summary\n' + '\n\n'.join(lines[-4:-1])
    return result


def main():
    parser = CommandLineParser(name='MuProver Training Client', game=True, replay_buffer=True)
    parser.add_argument('--logdir', type=str, metavar='PATH', required=False,
                        help='Directory for TensorBoard logging.')
    parser.add_argument('--min_games', type=int, default=1,
                        help='Minimum number of games required to start training')
    parser.add_argument('--saved_models', type=str, metavar='PATH', required=True,
                        help='Path to the models/ directory served by tensorflow serving.')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Set this flag to resume training from the latest checkpoint in --logdir directory.')
    parser.add_argument('--freeze-representation', action='store_true', default=False,
                        help='Set this flag to prevent training of the representation network.')
    parser.add_argument('--freeze-dynamics', action='store_true', default=False,
                        help='Set this flag to prevent training of the dynamics network.')
    parser.add_argument('--freeze-prediction', action='store_true', default=False,
                        help='Set this flag to prevent training of the prediction network.')
    args = parser.parse_args()

    if args.min_games < args.config.training_config.batch_size:
        parser.error(f'--min_games cannot be lower than the batch size {args.config.training_config.batch_size}')

    local_network = args.config.make_uniform_network()
    optimizer = args.config.training_config.optimizer
    checkpoint = tf.train.Checkpoint(network=local_network.checkpoint, optimizer=optimizer)
    checkpoint_path = os.path.join(args.logdir, 'ckpt') if args.logdir else None
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=None) if args.logdir else None

    if args.resume:
        if not checkpoint_manager:
            parser.error('A --logdir must be specified to --resume training from a checkpoint!')
        else:
            try:
                checkpoint_manager.restore_or_initialize()
            except tf.errors.NotFoundError:
                parser.error(f'unable to restore checkpoint from {checkpoint_path}!')
            else:
                print(f'Restored checkpoint from {checkpoint_path}!')

    if args.freeze_representation:
        local_network.representation.trainable = False
    if args.freeze_dynamics:
        local_network.dynamics.trainable = False
    if args.freeze_prediction:
        local_network.prediction.trainable = False

    writer = tf.summary.create_file_writer(args.logdir) if args.logdir else None
    if writer:
        hyperparameters = args.config.hyperparameters()
        with writer.as_default():
            hp.hparams(hyperparameters)
            tf.summary.text(name='Networks/Representation',
                            data=tensorboard_model_summary(local_network.representation),
                            step=0)
            tf.summary.text(name='Networks/Dynamics',
                            data=tensorboard_model_summary(local_network.dynamics),
                            step=0)
            tf.summary.text(name='Networks/Prediction',
                            data=tensorboard_model_summary(local_network.prediction),
                            step=0)
            tf.summary.text(name='Networks/Initial inference',
                            data=tensorboard_model_summary(local_network.initial_inference_model),
                            step=0)
            tf.summary.text(name='Networks/Recurrent inference',
                            data=tensorboard_model_summary(local_network.recurrent_inference_model),
                            step=0)

    if not os.path.isdir(args.saved_models):
        parser.error(f'--saved_models {args.saved_models} does not point to a valid directory!')
    local_network.save_tfx_models(args.saved_models)

    remote_replay_buffer = RemoteReplayBuffer(args.replay_buffer)
    try:
        remote_replay_buffer.stats()
    except RpcError:
        parser.error(f'Unable to connect to replay buffer at {args.replay_buffer}!')
    else:
        print(f'Connected to replay buffer at {args.replay_buffer}!')

    while remote_replay_buffer.num_games() < args.min_games:
        print(f'Waiting for {args.min_games} games to be available on the replay buffer...')
        time.sleep(60)

    train_network(config=args.config,
                  network=local_network,
                  optimizer=optimizer,
                  replay_buffer=remote_replay_buffer,
                  saved_models_path=args.saved_models,
                  writer=writer,
                  checkpoint_manager=checkpoint_manager)


if __name__ == '__main__':
    main()
