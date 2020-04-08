from storage_replay import *
from environment import *


class GameHistory:
    """
    Book-keeping class for completed games.
    """

    def __init__(self, action_space_size, num_players):
        self.action_space_size = action_space_size
        self.uniform_policy = [1 / action_space_size for _ in range(action_space_size)]
        self.players = [Player(i) for i in range(num_players)]
        self.action_list_hash = None

        self.observations = []
        self.actions = []
        self.rewards = []
        self.to_plays = []
        self.root_values = []
        self.policies = []

    def __str__(self):
        return 'Game({})'.format(', '.join(map(str, self.actions)))

    def __len__(self):
        return len(self.actions)

    def __hash__(self):
        if not self.action_list_hash:
            self.action_list_hash = sum(action.index * self.action_space_size**i for i, action in enumerate(self.actions))
        return self.action_list_hash

    def make_target(self, state_index, num_unroll_steps, td_steps, discount):
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value_dict = self.root_values[bootstrap_index]
            else:
                value_dict = {player: 0.0 for player in self.players}
            for player, value in value_dict.items():
                value_dict[player] = value * discount ** td_steps

            for i, reward_dict in enumerate(self.rewards[current_index:bootstrap_index]):
                for player, reward in reward_dict.items():
                    value_dict[player] = value_dict.get(player, 0.0) + reward * discount ** i

            # # For simplicity the network always predicts the most recently received
            # # reward, even for the initial representation network where we already
            # # know this reward.
            # if current_index > 0 and current_index <= len(self.rewards):
            #     last_reward = self.rewards[current_index - 1]
            # else:
            #     last_reward = self.no_value

            if current_index == state_index:
                last_reward = None
                to_play = None
            else:
                last_reward = [self.rewards[current_index-1].get(player, 0.0) for player in self.players]
                to_play = self.to_plays[current_index-1].player_id

            if current_index < len(self.root_values):
                targets.append(([value_dict.get(player, 0.0) for player in self.players],
                                last_reward,
                                self.policies[current_index],
                                to_play))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append(([0.0]*len(self.players), last_reward, self.uniform_policy, to_play))
        return targets


class Game:
    """
    A class to record episodes of interaction with an Environment.
    """

    def __init__(self, environment):
        self.environment = environment
        self.history = GameHistory(action_space_size=environment.action_space_size, num_players=environment.num_players)

    def to_play(self):
        return self.environment.to_play()

    def legal_actions(self):
        return self.environment.legal_actions()

    def terminal(self):
        return self.environment.terminal()

    def outcome(self):
        return self.environment.outcome()

    def apply(self, action):
        reward = self.environment.step(action)
        self.history.observations.append(self.make_image())
        self.history.actions.append(action)
        self.history.rewards.append({player: reward.get(player, 0.0) for player in self.environment.players()})
        self.history.to_plays.append(self.to_play())

    def store_search_statistics(self, root):
        action_space = map(Action, range(self.environment.action_space_size))
        self.history.policies.append([
            root.children[a].num_simulations / root.num_simulations if a in root.children else 0 for a in action_space
        ])
        self.history.root_values.append(
            {player: root.value_dict_sum.get(player, 0.0)/root.num_simulations for player in self.environment.players()}
        )

    def make_image(self):
        """
        Encode the state representation (returned by environment.get_state) into an np.array.
        """
        raise ImplementationError('make_image', 'Game')
