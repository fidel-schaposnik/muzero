import time, pickle, requests, json, tempfile, os
from mcts import *
from environment import *
from utils import *
from flask import Flask, send_file, request, send_from_directory


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
        return len(set(tuple(game_history.actions) for game_history in self.buffer))

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


class MuZeroServer:
    def __init__(self, config, data_path):
        # Set server directory and config
        self.server_dir = os.path.join(data_path, config.name, timestamp())
        os.makedirs(self.server_dir, exist_ok=True)

        with open(os.path.join(self.server_dir, 'config.pickle'), 'wb') as config_file:
            pickle.dump(config, config_file)
        self.config = config

        # Server statistics
        self.total_games = 0
        self.num_batches = 0
        self.num_unique = 0
        self.clients = {}
        self.start_time = datetime.datetime.now()

        # Shared storage
        self.shared_storage_dir = os.path.join(self.server_dir, 'shared_storage')
        os.makedirs(self.shared_storage_dir, exist_ok=True)

        self.network = self.config.make_uniform_network()
        self.network.save_weights(self.network_filepath_prefix(0))
        self.networks = {0}
        self.network_names = ('representation', 'dynamics', 'prediction')

        # Replay buffer
        self.replay_buffer_dir = os.path.join(self.server_dir, 'replay_buffer')
        os.makedirs(self.replay_buffer_dir, exist_ok=True)

        self.replay_buffer = ReplayBuffer(self.config)

        # Game playing
        self.game = self.config.new_game()

    def latest_step(self):
        return max(self.networks)

    def network_filepath_prefix(self, step):
        return os.path.join(self.shared_storage_dir, '{}_it{}'.format(self.config.name, step))

    def network_filepath(self, step, network_name):
        return '{}_{}.h5'.format(self.network_filepath_prefix(step), network_name[:3])

    def latest_network_filepath(self, network_name):
        if network_name not in self.network_names:
            return 'Invalid network requested!'

        return self.network_filepath(self.latest_step(), network_name)

    def register_client(self, client_id):
        self.clients[client_id] = datetime.datetime.now()

    def stats(self):
        return {'latest_step': self.latest_step(),
                'num_games': len(self.replay_buffer.buffer),
                'num_positions': self.replay_buffer.num_positions,
                'num_unique': self.num_unique,
                'total_games': self.total_games,
                'num_backups': self.total_games // self.replay_buffer.window_size,
                'num_batches': self.num_batches,
                'num_clients': sum(1 for time in self.clients.values() if (datetime.datetime.now() - time).total_seconds() < 3600),
                'total_runtime': str(datetime.datetime.now() - self.start_time)}

    def summary(self):
        with open('server/index.html', 'r') as template_file:
            template = template_file.read()
        for key, value in self.stats().items():
            template = template.replace(key, str(value))
        player_radiobuttons = '\n'.join('<tr><td><input type="radio" name="player" value="{}">{}</td></tr>'.format(player.player_id, player) for player in self.game.environment.players())
        template = template.replace('player_radiobuttons', player_radiobuttons)
        return template

    def save_network(self, step, weight_files):
        try:
            for network_name in self.network_names:
                weight_files[network_name].save(self.network_filepath(step, network_name))
            self.network.load_weights(self.network_filepath_prefix(step))
        except:
            return 'Server could not receive networks!'
        else:
            self.networks.add(step)
            return 'Server successfully received network at step {}!'.format(step)

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
        return 'Server successfully saved {} games!'.format(len(histories))

    def sample_batch(self):
        self.num_batches += 1
        return self.replay_buffer.sample_batch(self.config.num_unroll_steps, self.config.td_steps, self.config.discount)

    def show_game(self):
        with open('server/play.html', 'r') as template_file:
            template = template_file.read()

        template = template.replace('human_player', str(self.game.to_play().player_id))
        template = template.replace('current_state', self.game.state_repr().replace('\n', '<br>'))
        template = template.replace('to_play', str(self.game.to_play()))
        template = template.replace('finished', str(self.game.terminal()))
        template = template.replace('button_status', ' disabled' if self.game.terminal() else '')

        actions = []
        for action in self.game.legal_actions():
            action_row = '<input type="radio" name="move" value="{}">{}<br>'.format(action.index, action)
            actions.append(action_row)
        template = template.replace('legal_actions', '\n'.join(actions))

        history_rows = []
        for i, (action, reward) in enumerate(zip(self.game.history.actions, self.game.history.rewards)):
            state = self.game.state_repr(i).replace('\n', '<br>')
            row = '<tr><td>{}</td><td><tt>{}</tt></td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(i, state, self.game.history.to_plays[i-1] if i else Player(0), action, reward)
            history_rows.append(row)

        template = template.replace('history_rows', '\n'.join(history_rows))
        return template

    def play(self, player, action):
        if not action:
            self.game = self.config.new_game()
        else:
            self.game.apply(action)
        while not self.game.terminal() and len(self.game.history) < self.config.max_moves and self.game.to_play() != player:
            batch_make_move(self.config, self.network, [self.game], training=False)
            print('{} moves {}'.format(self.game.history.to_plays[-1], self.game.history.actions[-1]))
        return self.show_game()


class MuZeroClient:
    def __init__(self, server_address):
        self.server_address = server_address
        self.network_names = ('representation', 'dynamics', 'prediction')
        self.network_step = -1
        self.network = None

        self.client_id = random_id()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.client_id})
        print('Starting client with id={}...'.format(self.client_id))

        self.config = pickle.loads(self.session.get(server_address + 'config').content)
        print('Loaded configuration file for game {} from server {}!'.format(self.config.name, self.server_address))

    def stats(self):
        return self.session.get(self.server_address + 'json').json()

    def network_filepath_prefix(self, step):
        return os.path.join(tempfile.gettempdir(), '{}_it{}'.format(self.config.name, step))

    def network_filepath(self, step, network_name):
        return '{}_{}.h5'.format(self.network_filepath_prefix(step), network_name[:3])

    def latest_network(self):
        server_stats = self.stats()

        if self.network_step < server_stats['latest_step']:
            self.network_step = server_stats['latest_step']
            for network_name in self.network_names:
                response = self.session.get(self.server_address + 'storage/'+network_name)
                with open(self.network_filepath(self.network_step, network_name), 'wb') as tmp_file:
                    tmp_file.write(response.content)
            self.network = self.config.make_uniform_network()
            self.network.load_weights(self.network_filepath_prefix(self.network_step))
            self.network.steps = self.network_step

            for network_name in self.network_names:
                os.remove(self.network_filepath(self.network_step, network_name))
            print('Client successfully received network at step {}!'.format(self.network_step))
        return self.network

    def save_network(self, network):
        step = network.training_steps()
        network.save_weights(self.network_filepath_prefix(step))
        weight_files = {}
        for network_name in self.network_names:
            filepath = self.network_filepath(step, network_name)
            with open(filepath, 'rb') as file:
                weight_files[network_name] = file.read()
            os.remove(filepath)

        response = self.session.post(self.server_address+'storage', data={'step': step}, files=weight_files)
        print(response.text)

    def save_game_histories(self, histories):
        payload = pickle.dumps(histories)
        response = self.session.post(self.server_address + 'replay_buffer', files={'histories': payload})
        print(response.text)

    def sample_batch(self):
        response = self.session.get(self.server_address + 'replay_buffer')
        return pickle.loads(response.content)


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
        server.register_client(request.user_agent.string)
        step = int(request.form.get('step'))
        return server.save_network(step=step, weight_files=request.files)

    @api.route('/storage/<network_name>', methods=['GET'])
    def latest_network(network_name):
        server.register_client(request.user_agent.string)
        response = server.latest_network_filepath(network_name)
        if os.path.exists(response):
            return send_file(response, attachment_filename=os.path.basename(response))
        else:
            return response

    @api.route('/replay_buffer', methods=['GET', 'POST'])
    def save_game_histories():
        server.register_client(request.user_agent.string)
        if request.method == 'GET':
            payload = pickle.dumps(server.sample_batch())
            return payload
        else:
            game_histories = pickle.load(request.files['histories'])
            return server.save_game_histories(game_histories)

    @api.route('/play', methods=['GET', 'POST'])
    def play():
        if request.method == 'GET':
            return server.show_game()
        else:
            player = Player(int(request.form.get('player')))
            move = request.form.get('move')
            return server.play(player, Action(int(move)) if move else None)

    @api.route('/css/<path:path>')
    def send_css(path):
        return send_from_directory('server/css', path)

    @api.route('/config', methods=['GET'])
    def get_config():
        server.register_client(request.user_agent.string)
        return send_file(os.path.join(server.server_dir, 'config.pickle'), attachment_filename='config.pickle')

    api.run()
