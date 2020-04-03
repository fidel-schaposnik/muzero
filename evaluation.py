from mcts import *
from environment import *
import time


def batch_agent_duel(config, agents, games):
    while games:
        split_games = {agent: [] for agent in agents}
        for game in games:
            cur_player = game.to_play().player_id
            split_games[agents[cur_player]].append(game)

        for agent, cur_games in split_games.items():
            if agent == 'random':
                for game in cur_games:
                    action = random.choice(game.legal_actions())
                    game.apply(action)
            elif cur_games:
                batch_make_move(config, agent, cur_games)

        games = [game for game in games if not (game.terminal() or len(game.history) == config.max_moves)]


def evaluate_agent(config, agent, num_games):
    games = [config.new_game() for _ in range(num_games)]

    agents = ['random'] * len(games[0].environment.players())
    agents[-1] = agent

    start = time.time()
    batch_agent_duel(config, agents, games)
    end = time.time()
    print('Evaluation finished in {:.2f} seconds, {:.2f} seconds per game!'.format(end-start, (end-start)/num_games))

    stats = {}
    for game in games:
        for player, result in game.outcome().items():
            stats[player] = stats.get(player, 0.0) + result
    for player in stats.keys():
        stats[player] /= num_games
    print('Evaluation results: {}'.format(stats))
    return stats


def play_against_network(config, network, human_player_id=None, verbose=True):
    game = config.new_game()
    while not game.terminal() and len(game.history) < config.max_moves:
        if game.environment.to_play().player_id == human_player_id:
            action = Action(int(input('Input your action: ')))
            game.apply(action)
        else:
            batch_make_move(config, network, [game])
            if verbose:
                print('MuZero policy: {}'.format(game.history.policies[-1]))
                print('MuZero plays: {}'.format(game.history.actions[-1].index))
        if verbose:
            print('Reward: {}'.format(game.history.rewards[-1]))
            game.environment.print_state()
