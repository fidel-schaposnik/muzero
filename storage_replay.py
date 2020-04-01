from mcts import *
import time


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

    def save_game(self, game):
        if len(self.buffer) == self.window_size:
            self.num_positions -= len(self.buffer.pop(0))
        self.buffer.append(game.history)
        self.num_positions += len(game.history)

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


class SharedStorage:
    """
    Storage space for networks during training.
    """
    def __init__(self, config):
        self.networks = {}
        self.game_params = config.game_params
        self.network_class = config.network_class

    def latest_network(self):
        if not self.networks:
            self.networks[0] = self.make_uniform_network(**self.game_params)
        return self.networks[max(self.networks.keys())]

    def save_network(self, step, network):
        self.networks[step] = network

    def make_uniform_network(self, **game_params):
        return self.network_class(**game_params)


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
