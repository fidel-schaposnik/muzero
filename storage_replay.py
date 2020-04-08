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

    def num_unique(self):
        return len(set(self.buffer))

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
        os.makedirs(self.shared_storage_dir, exist_ok=True)

        self.network_names = ['representation', 'dynamics', 'prediction']
        self.network = config.make_uniform_network()
        self.network.save_weights(self.network_filepath_prefix(0))
        self.networks = {0}

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config)

        self.replay_buffer_dir = os.path.join(self.server_dir, 'replay_buffer')
        os.makedirs(self.replay_buffer_dir, exist_ok=True)

        self.total_games = 0
        self.num_batches = 0
        self.num_unique = 0

    def latest_step(self):
        return max(self.networks)

    def network_filepath_prefix(self, step):
        return os.path.join(self.shared_storage_dir, '{}_it{}'.format(self.name, step))

    def network_filepath(self, step, network_name):
        return '{}_{}.h5'.format(self.network_filepath_prefix(step), network_name[:3])

    def latest_network_filepath(self, network_name):
        if network_name not in self.network_names:
            return 'Invalid network requested!'

        return self.network_filepath(self.latest_step(), network_name)

    def stats(self):
        return {'latest_step': self.latest_step(),
         'num_games': len(self.replay_buffer.buffer),
         'num_positions': self.replay_buffer.num_positions,
         'num_unique': self.num_unique,
         'total_games': self.total_games,
         'num_backups': self.total_games // self.replay_buffer.window_size,
         'num_batches': self.num_batches}

    def summary(self):
        with open('index.html', 'r') as template_file:
            template = template_file.read()
        for key, value in self.stats().items():
            template = template.replace(key, str(value))
        return template

    def save_network(self, step, weight_files):
        try:
            for network_name in self.network_names:
                weight_files[network_name].save(self.network_filepath(step, network_name))
            self.network.load_weights(self.network_filepath_prefix(step))
        except:
            return 'Could not receive networks!'
        else:
            self.networks.add(step)
            return 'Successfully received networks at step {}!'.format(step)

    def backup_filepath(self):
        num_backups = self.total_games//self.replay_buffer.window_size
        filename = '{}_game_histories_{}.pickle'.format(self.replay_buffer.window_size, num_backups)
        return os.path.join(self.replay_buffer_dir, filename)

    def backup_histories(self):
        with open(self.backup_filepath(), 'wb') as backup_file:
            pickle.dump(self.replay_buffer.buffer, backup_file)

    def save_game_histories(self, histories):
        for history in histories:
            self.replay_buffer.save_history(history)
            self.total_games += 1
            if self.total_games % self.replay_buffer.window_size == 0:
                self.backup_histories()
        self.num_unique = self.replay_buffer.num_unique()
        return 'Successfully saved {} games!'.format(len(histories))

    def sample_batch(self):
        self.num_batches += 1
        return self.replay_buffer.sample_batch(self.num_unroll_steps, self.td_steps, self.discount)


class MuZeroClient:
    def __init__(self, config, server_address):
        self.name = config.name
        self.make_uniform_network = config.make_uniform_network
        self.server_address = server_address
        self.network_names = ['representation', 'dynamics', 'prediction']
        self.network_step = -1
        self.network = None

    def stats(self):
        return json.loads(requests.get(self.server_address + 'json').text)

    def network_filepath_prefix(self, step):
        return os.path.join(tempfile.gettempdir(), '{}_it{}'.format(self.name, step))

    def network_filepath(self, step, network_name):
        return '{}_{}.h5'.format(self.network_filepath_prefix(step), network_name[:3])

    def latest_network(self):
        server_stats = self.stats()

        if self.network_step < server_stats['latest_step']:
            self.network_step = server_stats['latest_step']
            for network_name in self.network_names:
                response = requests.get(self.server_address + 'storage/'+network_name)
                with open(self.network_filepath(self.network_step, network_name), 'wb') as tmp_file:
                    tmp_file.write(response.content)

            self.network = self.make_uniform_network()
            self.network.load_weights(self.network_filepath_prefix(self.network_step))
            self.network.steps = self.network_step

            for network_name in self.network_names:
                os.remove(self.network_filepath(self.network_step, network_name))
        return self.network

    def save_network(self, network):
        step = network.training_steps()
        network.save_weights(self.network_filepath_prefix(step))
        weight_files = {}
        for network_name in self.network_names:
            filepath = self.network_filepath(step, network_name)
            with open(filepath, 'rb') as file :
                weight_files[network_name] = file.read()
            os.remove(filepath)

        response = requests.post(self.server_address+'storage', data={'step': step}, files=weight_files)
        print(response.text)

    def save_game_histories(self, histories):
        payload = pickle.dumps(histories)
        response = requests.post(self.server_address + 'replay_buffer', files={'histories': payload})
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
    def server_summary():
        return server.summary()

    @api.route('/json', methods=['GET'])
    def server_stats():
        return json.dumps(server.stats())

    @api.route('/storage', methods=['POST'])
    def save_network():
        step = int(request.form.get('step'))
        return server.save_network(step=step, weight_files=request.files)

    @api.route('/storage/<network_name>', methods=['GET'])
    def latest_network(network_name):
        response = server.latest_network_filepath(network_name)
        if os.path.exists(response):
            return send_file(response, attachment_filename=os.path.basename(response))
        else:
            return response

    @api.route('/replay_buffer', methods=['GET', 'POST'])
    def save_game_histories():
        if request.method == 'GET':
            payload = pickle.dumps(server.sample_batch())
            return payload
        else:
            game_histories = pickle.load(request.files['histories'])
            return server.save_game_histories(game_histories)

    api.run()
