from grpc import RpcError

from muzero.agent import MCTSAgent
from muzero.utils import CommandLineParser
from muzero.replay_buffer_services import RemoteReplayBuffer
from muzero.network_services import RemoteNetwork
from muzero.environment_services import RemoteEnvironment


def main():
    parser = CommandLineParser(name='MuProver MCTSAgent', game=True, replay_buffer=True, network=True, environment=True)
    parser.add_argument('--num_games', type=int, default=int(1e10),
                        help='Number of games to play (defaults to infinity)')
    parser.add_argument('--name', type=str, help='A name for this agent.')
    parser.add_argument('--temperature', type=float, help='Temperature for the softmax distribution of action choices.')
    args = parser.parse_args()

    if args.num_games <= 0:
        parser.error('--num_games must be a strictly positive integer number!')

    if args.temperature:
        if args.temperature < 0:
            parser.error('--temperature must be a non-negative floating-point number!')
        args.config.mcts_config.temperature = args.temperature

    try:
        remote_network = RemoteNetwork(config=args.config, ip_port=args.network)
    except RpcError:
        parser.error(f'Unable to connect to remote network at {args.network}!')
        remote_network = None
    else:
        print(f'Connected to remote network at {args.network}!')

    agent = MCTSAgent(config=args.config, network=remote_network, agent_id=args.name)

    remote_replay_buffer = RemoteReplayBuffer(args.replay_buffer)
    try:
        remote_replay_buffer.stats()
    except RpcError:
        parser.error(f'Unable to connect to replay buffer at {args.replay_buffer}!')
    else:
        print(f'Connected to replay buffer at {args.replay_buffer}!')

    with RemoteEnvironment(config=args.config, ip_port=args.environment) as remote_environment:
        print(f'Connected to remote environment at {args.environment}!')

        for _ in range(args.num_games):
            game_history = agent.play_game(remote_environment)

            timing = game_history.metadata['timing']
            num_moves = len(game_history)
            print(f'Game played in {timing:.2f}s, {num_moves} moves ==> {timing / num_moves:.2f}s per move!')
            print(game_history)
            remote_replay_buffer.save_history(game_history)


if __name__ == '__main__':
    main()
