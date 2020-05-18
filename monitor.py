from flask import Flask, send_file, request, send_from_directory
import argparse

from replay_buffer import RemoteReplayBuffer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuZero MCTS Agent')
    parser.add_argument('--replay_buffer', type=str, required=True,
                        help='IP:Port for gRPC communication with a replay buffer server')
    # parser.add_argument('--environment', type=str, required=True,
    #                     help='IP:Port for gRPC communication with an environment server')
    # parser.add_argument('--network', type=str, required=True,
    #                     help='IP:Port for gRPC communication with a network server')
    args = parser.parse_args()
    remote_replay_buffer = RemoteReplayBuffer(args.replay_buffer)


    api = Flask(__name__)

    @api.route('/css/<path:path>')
    def stylesheet(path):
        return send_from_directory('monitoring_server/css', path)

    @api.route('/', methods=['GET'])
    def summary():
        stats = remote_replay_buffer.stats()
        with open('monitoring_server/index.html', 'r') as template_file:
            template = template_file.read()
        for key, value in stats.items():
            template = template.replace(key, '{:.2f}'.format(value))
        return template

    # @api.route('/json', methods=['GET'])
    # def server_stats():
    #     return json.dumps(server.stats())
    #
    # @api.route('/storage', methods=['POST'])
    # def save_network():
    #     server.register_client(request.user_agent.string)
    #     step = int(request.form.get('step'))
    #     return server.save_network(step=step, weight_files=request.files)
    #
    # @api.route('/storage/<network_name>', methods=['GET'])
    # def latest_network(network_name):
    #     server.register_client(request.user_agent.string)
    #     response = server.latest_network_filepath(network_name)
    #     if os.path.exists(response):
    #         return send_file(response, attachment_filename=os.path.basename(response))
    #     else:
    #         return response
    #
    # @api.route('/replay_buffer', methods=['GET', 'POST'])
    # def save_game_histories():
    #     server.register_client(request.user_agent.string)
    #     if request.method == 'GET':
    #         payload = pickle.dumps(server.sample_batch())
    #         return payload
    #     else:
    #         game_histories = pickle.load(request.files['histories'])
    #         return server.save_game_histories(game_histories)
    #
    # @api.route('/play', methods=['GET', 'POST'])
    # def play():
    #     if request.method == 'GET':
    #         return server.show_game()
    #     else:
    #         player = Player(int(request.form.get('player')))
    #         move = request.form.get('move')
    #         return server.play(player, Action(int(move)) if move else None)
    #
    #
    # @api.route('/config', methods=['GET'])
    # def get_config():
    #     server.register_client(request.user_agent.string)
    #     return send_file(os.path.join(server.server_dir, 'config.pickle'), attachment_filename='config.pickle')

    api.run()
