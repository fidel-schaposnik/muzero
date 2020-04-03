import time, pickle, requests, json, tempfile, os
from mcts import *
from utils import timestamp
from flask import Flask, send_file, request


class ReplayBuffer:
    """
    Buffer where games played by MuZero are stored for training purposes.
    """
    def __init__(self, config):
        self.window_size = config.window_size
        self.buffer = []
        self.num_positions = 0

        self.batch_size = config.batch_size
        self.num_unroll_steps = config.num_unroll_steps

    def save_history(self, history):
        if len(self.buffer) == self.window_size:
            self.num_positions -= len(self.buffer.pop(0))
        self.buffer.append(history)
        self.num_positions += len(history)

    def save_game(self, game):
        self.save_history(game.history)

    def sample_batch(self, num_unroll_steps, td_steps, discount):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.observations[i],
                 g.actions[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, discount)) for (g, i) in game_pos]

    def sample_game(self):
        return random.choice(self.buffer)

    def sample_position(self, game_history) -> int:
        return random.randrange(len(game_history)-self.num_unroll_steps+1)


# class SharedStorage:
#     """
#     Storage space for networks during training.
#     """
#     def __init__(self, config):
#         self.networks = {}
#         self.game_params = config.game_params
#         self.network_class = config.network_class
#
#     def latest_network(self):
#         if not self.networks:
#             self.networks[0] = self.make_uniform_network(**self.game_params)
#         return self.networks[max(self.networks.keys())]
#
#     def save_network(self, step, network):
#         self.networks[step] = network


class MuZeroServer:
    def __init__(self, config, data_path):
        self.name = config.name
        self.num_unroll_steps = config.num_unroll_steps
        self.td_steps = config.td_steps
        self.discount = config.discount
        # self.config_hash = config.hash()

        self.server_dir = os.path.join(data_path, self.name, timestamp())

        # Shared storage
        self.shared_storage_dir = os.path.join(self.server_dir, 'shared_storage')
        if not os.path.exists(self.shared_storage_dir):
            os.makedirs(self.shared_storage_dir)

        self.network_names = {'representation', 'dynamics', 'prediction'}
        config.make_uniform_network().save_weights(self.network_filepath_prefix(0))
        self.networks = {0: self.network_names}

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config)

        self.replay_buffer_dir = os.path.join(self.server_dir, 'replay_buffer')
        if not os.path.exists(self.replay_buffer_dir):
            os.makedirs(self.replay_buffer_dir)

        self.to_backup = 0
        self.num_backups = 0
        self.num_batches = 0

    def latest_step(self):
        return max(step for step, nameset in self.networks.items() if nameset == self.network_names)

    def network_filepath_prefix(self, step):
        return os.path.join(self.shared_storage_dir, '{}_it{}'.format(self.name, step))

    def network_filepath(self, step, network_name):
        return '{}_{}.h5'.format(self.network_filepath_prefix(step), network_name[:3])

    def latest_network_filepath(self, network_name):
        if network_name not in self.network_names:
            return 'Invalid network requested!'

        return self.network_filepath(self.latest_step(), network_name)

    def information(self):
        return json.dumps({'latest_step': self.latest_step(),
                           'num_games': len(self.replay_buffer.buffer),
                           'num_positions': self.replay_buffer.num_positions,
                           'num_backups': self.num_backups,
                           'num_batches': self.num_batches
                           })

    def save_network(self, step, network_name, payload):
        if network_name not in self.network_names:
            return 'Invalid network requested!'

        weights_file = open(self.network_filepath(step, network_name), 'wb')
        weights_file.write(payload)
        weights_file.close()

        self.networks.setdefault(step, set()).add(network_name)
        return 'Successfully received {} network at step {}!'.format(network_name, step)

    def backup_filepath(self):
        return os.path.join(self.replay_buffer_dir, '{}_game_histories_{}.pickle'.format(self.replay_buffer.window_size, self.num_backups))

    def backup_histories(self):
        backup_file = open(self.backup_filepath(), 'wb')
        pickle.dump(self.replay_buffer.buffer, backup_file)
        backup_file.close()
        self.to_backup = 0
        self.num_backups += 1

    def save_game_histories(self, histories):
        for history in histories:
            self.replay_buffer.save_history(history)
            self.to_backup += 1
            if self.to_backup == self.replay_buffer.window_size:
                self.backup_histories()
        return 'Successfully saved {} games!'.format(len(histories))

    def sample_batch(self):
        self.num_batches += 1
        return self.replay_buffer.sample_batch(self.num_unroll_steps, self.td_steps, self.discount)


class MuZeroClient:
    def __init__(self, config, server_address):
        self.name = config.name
        self.make_uniform_network = config.make_uniform_network
        self.server_address = server_address
        self.network_names = {'representation', 'dynamics', 'prediction'}
        self.network_step = -1
        self.network = None

    def network_filepath_prefix(self, step):
        return os.path.join(tempfile.gettempdir(), '{}_it{}'.format(self.name, step))

    def network_filepath(self, step, network_name):
        return '{}_{}.h5'.format(self.network_filepath_prefix(step), network_name[:3])

    def latest_network(self):
        server_information = json.loads(requests.get(self.server_address).text)

        if self.network_step < server_information['latest_step']:
            self.network_step = server_information['latest_step']
            for network_name in self.network_names:
                response = requests.get(self.server_address + 'storage/'+network_name)
                tmp_file = open(self.network_filepath(self.network_step, network_name), 'wb')
                tmp_file.write(response.content)
                tmp_file.close()
            self.network = self.make_uniform_network()
            self.network.load_weights(self.network_filepath_prefix(self.network_step))
            self.network.steps = self.network_step
        return self.network

    def save_network(self, network):
        network.save_weights(self.network_filepath_prefix(network.training_steps()))
        for network_name in self.network_names:
            tmp_file = open(self.network_filepath(network.training_steps(), network_name), 'rb')
            payload = tmp_file.read()
            tmp_file.close()
            response = requests.post(self.server_address + 'storage/'+network_name, params={'step': network.training_steps()}, data=payload)
            print(response.text)

    def save_game_histories(self, histories):
        payload = pickle.dumps(histories)
        response = requests.post(self.server_address + 'replay_buffer', data=payload)
        print(response.text)

    def sample_batch(self):
        response = requests.get(self.server_address + 'replay_buffer')
        return pickle.loads(response.content)


def batch_selfplay(config, replay_buffer, network, num_games):
    start = time.time()
    games = [config.new_game() for _ in range(num_games)]
    while games:
        batch_make_move(config, network, games)

        open_games = []
        for game in games:
            if game.terminal() or len(game.history) == config.max_moves:
                replay_buffer.save_game(game)
            else:
                open_games.append(game)
        games = open_games
    end = time.time()
    print('Generated {} games in {:.2f} seconds, {:.2f} per game!'.format(num_games, end-start, (end-start)/num_games))


def start_server(config, data_path):
    server = MuZeroServer(config, data_path)

    api = Flask(__name__)

    @api.route('/', methods=['GET'])
    def server_information():
        return server.information()

    @api.route('/storage/<network_name>', methods=['GET', 'POST'])
    def storage(network_name):
        if request.method == 'GET':
            response = server.latest_network_filepath(network_name)
            if os.path.exists(response):
                return send_file(response, attachment_filename=os.path.basename(response))
            else:
                return response
        else:
            step = int(request.args.get('step'))
            return server.save_network(step, network_name, request.data)

    @api.route('/replay_buffer', methods=['GET', 'POST'])
    def save_game_histories():
        if request.method == 'GET':
            payload = pickle.dumps(server.sample_batch())
            return payload
        else:
            game_histories = pickle.loads(request.data)
            return server.save_game_histories(game_histories)

    api.run()