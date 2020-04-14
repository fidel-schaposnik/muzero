from importlib import import_module
from training import *
# from evaluation import *
# from training import *
import argparse


def load_game(game_name):
    try:
        game_module = import_module('games.{}'.format(game_name))
    except ModuleNotFoundError:
        parser.error('Choice of --game {} is invalid!'.format(game_name))
    else:
        return game_module.make_config()


parser = argparse.ArgumentParser(description='MuZero')
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument('--server', metavar='DATA_DIR', type=str, help='Start MuZero server for network storage and replay buffer')
mode_group.add_argument('--client', metavar='HOST', type=str, help='Start MuZero client for training or self-play')
mode_group.add_argument('--synchronous', help='Start MuZero in synchronous mode', action='store_true')
parser.add_argument('--game', type=str, help='One of the games implemented in the games/ directory')
type_group = parser.add_mutually_exclusive_group()
type_group.add_argument('--train', metavar='NUM_EVAL_GAMES', type=int, help='MuZero training client playing NUM_EVAL_GAMES evaluation games at each checkpoint')
type_group.add_argument('--self-play', metavar='NUM_GAMES', type=int, help='MuZero self-playing client generating batches of NUM_GAMES games')
parser.add_argument('--num-steps', type=int, help='Number of training steps per cycle')
parser.add_argument('--num-games', type=int, help='Number of self-play games per cycle')
parser.add_argument('--num-eval-games', type=int, help='Number of evaluation games to play at each checkpoint')
args = parser.parse_args()

if args.server:
    if not args.game:
        parser.error('You need to specify the --game for the --server in asynchronous mode.')
    if args.train or args.self_play:
        parser.error('--train and --self-play can only be set for a --client in asynchronous mode.')
    elif args.num_steps or args.num_games or args.num_eval_games:
        parser.error('--num-steps, --num-games and --num-eval-games can only be set in --synchronous mode.')
    else:
        config = load_game(args.game)
        if config:
            start_server(config, args.server)
elif args.client:
    if not args.train and not args.self_play:
        parser.error('--client needs to be set to --train or --self-play.')
    elif args.num_steps or args.num_games or args.num_eval_games:
        parser.error('--num-steps, --num-games and --num-eval-games can only be set in --synchronous mode.')
    elif args.train:
        train_network(args.client, args.train, tensorboard_logpath='checkpoints')
    elif args.self_play:
        client = MuZeroClient(args.client)
        while True:
            network = client.latest_network()
            replay_buffer = ReplayBuffer(client.config)
            batch_selfplay(client.config, replay_buffer, network, args.self_play)
            client.save_game_histories(replay_buffer.buffer)
elif args.synchronous:
    if not args.game:
        parser.error('You need to specify the --game for --synchronous mode.')
    if not args.num_eval_games:
        args.num_eval_games = 0
    if not args.num_steps or not args.num_games:
        parser.error('Setting --num-steps and --num-games is required in --synchronous mode.')
    elif args.train or args.self_play:
        parser.error('--train and --self-play can only be set in --client mode.')
    else:
        config = load_game(args.game)
        if config:
            network = config.make_uniform_network()

            synchronous_train_network(config, network,
                                      num_games=args.num_games,
                                      num_steps=args.num_steps,
                                      num_eval_games=args.num_eval_games,
                                      checkpoint_path='checkpoints')
