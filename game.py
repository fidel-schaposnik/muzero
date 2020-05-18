from environment import Action


class GameHistory:
    """
    Book-keeping class for completed games.
    """

    def __init__(self):
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

    def compute_target_value(self, index, td_steps, discount):
        bootstrap_index = index + td_steps
        if bootstrap_index < len(self.root_values):
            value = self.root_values[bootstrap_index] * discount ** td_steps
        else:
            value = 0

        for i, reward in enumerate(self.rewards[index:bootstrap_index]):
            sign = 1 if self.to_plays[index+i] == self.to_plays[index] else -1
            value += sign * reward * discount ** i
        return value

    def make_target(self, state_index, num_unroll_steps, td_steps, discount):
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            value = self.compute_target_value(current_index, td_steps, discount)

            if 0 < current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = 0

            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.policies[current_index]))
            else:
                targets.append((value, last_reward, [1/len(self.policies[-1])]*len(self.policies[-1])))
        return targets


class Game:
    """
    A class to record episodes of interaction with an Environment.
    """

    def __init__(self, environment):
        self.environment = environment
        self.history = GameHistory()

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
        self.history.rewards.append(reward)
        self.history.to_plays.append(self.to_play())

    def store_search_statistics(self, root):
        action_space = map(Action, range(self.environment.action_space_size))
        self.history.policies.append([
            root.children[a].visit_count / root.visit_count if a in root.children else 0 for a in action_space
        ])
        self.history.root_values.append(float(root.value()))

    def make_image(self):
        """
        If necessary, encode the state representation (returned by environment.get_state) into an np.array.
        """
        return self.environment.get_state()
