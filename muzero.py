from importlib import import_module
from training import *
# from evaluation import *
# from training import *
import argparse

parser = argparse.ArgumentParser(description='MuZero')
parser.add_argument('--game', type=str, help='One of the games implemented in games/ directory', required=True)
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument('--server', metavar='DATA_DIR', type=str, help='Start MuZero server for network storage and replay buffer')
mode_group.add_argument('--client', metavar='HOST', type=str, help='Start MuZero client for training or self-play')
mode_group.add_argument('--synchronous', help='Start MuZero client for training or self-play', action='store_true')
type_group = parser.add_mutually_exclusive_group()
type_group.add_argument('--train', metavar='NUM_EVAL_GAMES', type=int, help='MuZero training agent')
type_group.add_argument('--self-play', metavar='NUM_GAMES', type=int, help='MuZero self-playing agent')
parser.add_argument('--num-steps', type=int, help='Number of training steps per cycle')
parser.add_argument('--num-games', type=int, help='Number of self-play games per cycle')
parser.add_argument('--num-eval-games', type=int, help='Number of evaluation games to play at each checkpoint.')
args = parser.parse_args()


try:
    game_module = import_module('games.{}'.format(args.game))
except ModuleNotFoundError:
    parser.error('Choice of --game {} is invalid!'.format(args.game))
    exit(0)
else:
    config = game_module.make_config()

if args.server:
    if args.train or args.self_play:
        parser.error('--train and --self-play can only be set in --client mode.')
    elif args.num_steps or args.num_games or args.num_eval_games:
        parser.error('--num-steps, --num-games and --num-eval-games can only be set in --synchronous mode.')
    else:
        start_server(config, args.server)
elif args.client:
    if not args.train and not args.self_play:
        parser.error('--client needs to be set to --train or --self-play.')
    elif args.num_steps or args.num_games or args.num_eval_games:
        parser.error('--num-steps, --num-games and --num-eval-games can only be set in --synchronous mode.')
    elif args.train:
        train_network(config, args.client, args.train, tensorboard_logpath='checkpoints')
    elif args.self_play:
        client = MuZeroClient(config, args.client)
        while True:
            network = client.latest_network()
            replay_buffer = ReplayBuffer(config)
            batch_selfplay(config, replay_buffer, network, args.self_play)
            client.save_game_histories(replay_buffer.buffer)
elif args.synchronous:
    if not args.num_steps or not args.num_games or not args.num_eval_games:
        parser.error('Setting --num-steps, --num-games and --num-eval-games is required in --synchronous mode.')
    elif args.train or args.self_play:
        parser.error('--train and --self-play can only be set in --client mode.')
    else:
        network = config.make_uniform_network()
        synchronous_train_network(config, network,
                                  num_games=args.num_games,
                                  num_steps=args.num_steps,
                                  num_eval_games=args.num_eval_games,
                                  checkpoint_path='checkpoints')

# config = make_tictactoe_config()
# network = config.make_uniform_network()
# replay_buffer = ReplayBuffer(config)
#
# client.save_game_histories(replay_buffer.buffer)
# batch = client.sample_batch()
# print(batch)


# network = storage.latest_network()
# network.load_weights(r'checkpoints\TicTacToe\01-04-2020--15-16\TicTacToe_it6000')
# evaluate_agent(config, network, num_games=10)
# play_against_network(config, network, human_player_id=1)
#
# batch_selfplay(config, replay_buffer, network, num_games=1)
# print(replay_buffer.buffer[0])
#
# batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps, config.discount)
# print(batch)
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
# synchronous_train_network(config, network, num_games=100, num_steps=100, num_eval_games=10, checkpoint_path='checkpoints')

# game = config.new_game()
# game.apply(Action(4))
# state = np.expand_dims(game.make_image(),0)
# print(state.shape)
# hidden_state = network.representation(state)
# print(network.prediction(hidden_state))
