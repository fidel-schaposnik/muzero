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

    def __init__(self, action_space_size, num_players):
        self.action_space_size = action_space_size
        self.num_players = num_players

    def players(self):
        """
        Returns a list of all Player's acting in the environment.
        """
        return [Player(player_id) for player_id in range(self.num_players)]

    def to_play(self):
        """
        Returns the Player that is next to play.
        """
        raise ImplementationError('to_play', 'Environment')

    def is_legal_action(self, action):
        """
        Returns True if the action is legal in the current state, False otherwise.
        """
        raise ImplementationError('is_legal_action', 'Environment')

    def legal_actions(self):
        """
        Returns a list of all Action's that can be performed in the current state.
        """
        return [action for action in map(Action, range(self.action_space_size)) if self.is_legal_action(action)]

    def terminal(self):
        """
        Returns True if the game has ended, False otherwise.
        """
        raise ImplementationError('terminal', 'Environment')

    def outcome(self):
        """
        Returns a dictionary with a numerical value for each Player at the end of the game.
        """
        raise ImplementationError('outcome', 'Environment')

    def step(self, action):
        """
        Performs an Action: updates the internal state and returns a dictionary of rewards for each Player in the game.
        """
        raise ImplementationError('step', 'Environment')

    def get_state(self):
        """
        Returns a canonical representation of the environment's current state (encoding is done in game.make_image).
        """
        raise ImplementationError('get_state', 'Environment')
