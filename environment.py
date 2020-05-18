import tensorflow as tf
import grpc
import logging
import argparse
from concurrent import futures

from exceptions import *
from utils import load_game
from protos import environment_pb2
from protos import environment_pb2_grpc


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
        return self.player_id

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
    Sub-class this to implement your own environments.
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


class RemoteEnvironmentServer(environment_pb2_grpc.RemoteEnvironmentServicer):
    """
    A server for local environments, exposing them through a common gRPC API.
    """
    def __init__(self, environment_class):
        self.environments = {}
        self.environment_class = environment_class
        self.environment_id = 0

    @staticmethod
    def fill_state(state, environment):
        state.observation.CopyFrom(tf.make_tensor_proto(environment.get_state()))
        state.to_play = environment.to_play().player_id
        state.legal_actions.extend([action.index for action in environment.legal_actions()])

    def Initialization(self, request, context):
        environment = self.environment_class(**request.environment_parameters)
        self.environments[self.environment_id] = environment

        response = environment_pb2.InitializationResponse(environment_id=self.environment_id,
                                                          action_space_size=environment.action_space_size,
                                                          num_players=environment.num_players)
        self.fill_state(state=response.state, environment=environment)
        self.environment_id += 1
        return response

    def Step(self, request, context):
        response = environment_pb2.ActionResponse()
        try:
            environment = self.environments[request.environment_id]
            reward = environment.step(Action(request.index))
        except KeyError or AssertionError:
            response.success = False
        else:
            response.success = True
            response.reward = reward
            self.fill_state(state=response.state, environment=environment)
            response.done = environment.terminal()

            if environment.terminal():
                del(self.environments[request.environment_id])
        return response


class RemoteEnvironment(Environment):
    """
    Connects to an environment server and interacts with it.
    Behaves exactly like Environment, but is agnostic about how the server deals with the environment.
    """
    def __init__(self, game_config, ip_port):
        channel = grpc.insecure_channel(ip_port)
        self.remote_environment = environment_pb2_grpc.RemoteEnvironmentStub(channel)

        request = environment_pb2.InitializationRequest(environment_parameters=game_config.environment_parameters)
        response = self.remote_environment.Initialization(request)

        super().__init__(action_space_size=response.action_space_size, num_players=response.num_players)
        self.environment_id = response.environment_id
        self.state = response.state
        self.ended = False

    def to_play(self):
        return Player(self.state.to_play)

    def is_legal_action(self, action):
        return action.index in self.state.legal_actions

    def legal_actions(self):
        return [Action(index) for index in self.state.legal_actions]

    def terminal(self):
        return self.ended

    def step(self, action):
        request = environment_pb2.ActionRequest(environment_id=self.environment_id, index=action.index)
        response = self.remote_environment.Step(request)
        assert response.success

        self.state = response.state
        self.ended = response.done
        return response.reward

    def get_state(self):
        return tf.make_ndarray(self.state.observation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuZero Environment Server')
    parser.add_argument('--game', type=str, help='One of the games implemented in the games/ directory', required=True)
    parser.add_argument('--port', type=str, help='Port for gRPC communication', required=True)
    args = parser.parse_args()

    def serve(game_config, port):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        servicer = RemoteEnvironmentServer(game_config.environment_class)
        environment_pb2_grpc.add_RemoteEnvironmentServicer_to_server(servicer, server)
        server.add_insecure_port('[::]:{}'.format(port))
        print('Starting server for environment {} in port {}...'.format(game_config.name, port))
        server.start()
        server.wait_for_termination()

    config = load_game(args.game, parser)

    logging.basicConfig()
    serve(config.game_config, args.port)
