from exceptions import *

class Player:
    """
    A class for players: for two-player games, player_id is 0 or 1 and player 0 starts.
    """

    def __init__(self, player_id):
        self.player_id = player_id

    def __repr__(self):
        return 'Player({})'.format(self.player_id)

    def __eq__(self, other):
        return self.player_id == other.player_id

    def __hash__(self):
        return int(self.player_id)  # player_id may be a tf.Tensor coming from neural network evaluation

    def __gt__(self, other):
        return self.player_id > other.player_id


class Action:
    """
    A class for actions.
    """

    def __init__(self, index: int):
        self.index = index

    def __repr__(self):
        return 'Action({})'.format(self.index)

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index

class Environment:
    """
    A class for environments with which MuZero interacts.
    """

    def players(self):
        raise ImplementationError('players', 'Environment')

    def to_play(self):
        raise ImplementationError('to_play', 'Environment')

    def legal_actions(self):
        raise ImplementationError('legal_actions', 'Environment')

    def terminal(self):
        raise ImplementationError('terminal', 'Environment')

    def outcome(self):
        raise ImplementationError('outcome', 'Environment')

    def step(self, action):
        raise ImplementationError('step', 'Environment')

    def get_state(self):
        raise ImplementationError('get_state', 'Environment')