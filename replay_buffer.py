import logging
import grpc
import random
import tensorflow as tf
import argparse
from concurrent import futures

from protos import replay_buffer_pb2_grpc
from protos import replay_buffer_pb2
from game import *
from utils import load_game
from environment import Action, Player


class ReplayBuffer:
    """
    Buffer where games played by MuZero are stored for training purposes.
    """
    def __init__(self, replay_buffer_config):
        self.window_size = replay_buffer_config.window_size
        self.buffer = []
        self.num_positions = 0
        self.total_games = 0

    @staticmethod
    def sample_position(game_history, num_unroll_steps):
        return random.randrange(len(game_history)-num_unroll_steps+1)

    def save_history(self, history):
        if len(self.buffer) == self.window_size:
            self.num_positions -= len(self.buffer.pop(0))
        self.buffer.append(history)
        self.num_positions += len(history)
        self.total_games += 1

    def save_game(self, game):
        self.save_history(game.history)

    def sample_batch(self, batch_size, num_unroll_steps, td_steps, discount):
        games = [self.sample_game() for _ in range(batch_size)]
        game_pos = [(g, self.sample_position(g, num_unroll_steps)) for g in games]
        return [(g.observations[i],
                 g.actions[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, discount)) for (g, i) in game_pos]

    def sample_game(self):
        return random.choice(self.buffer)

    def stats(self):
        positions = set()
        for game_history in self.buffer:
            positions.update([tuple(game_history.actions[:i+1]) for i in range(len(game_history))])
        metrics = {
            'num_games': len(self.buffer),
            'num_unique_games': len(set(tuple(game_history.actions) for game_history in self.buffer)),
            'num_positions': self.num_positions,
            'num_unique_positions': len(positions),
            'total_games': self.total_games,
            'avg_game_length': self.num_positions/len(self.buffer)
        }
        return metrics


class ReplayBufferServer(replay_buffer_pb2_grpc.ReplayBufferServicer):
    """
    A server for replay buffers, exposing their functionality through a gRPC API
    """
    def __init__(self, replay_buffer_config):
        self.replay_buffer = ReplayBuffer(replay_buffer_config)

    def SaveGame(self, request, context):
        game_history = GameHistory()
        game_history.observations = [tf.make_ndarray(observation) for observation in request.observations]
        game_history.actions = [Action(index) for index in request.actions]
        game_history.rewards = request.rewards
        game_history.to_plays = [Player(player_id) for player_id in request.to_plays]
        game_history.root_values = request.root_values
        game_history.policies = [policy.probabilities for policy in request.policies]
        self.replay_buffer.save_history(game_history)
        print('Number of games in buffer: {}'.format(len(self.replay_buffer.buffer)))
        return replay_buffer_pb2.SaveGameResponse(success=True)

    def SampleBatch(self, request, context):
        batch = self.replay_buffer.sample_batch(batch_size=request.batch_size,
                                                num_unroll_steps=request.num_unroll_steps,
                                                td_steps=request.td_steps,
                                                discount=request.discount)

        response = replay_buffer_pb2.MiniBatchResponse()
        for observation, actions, targets in batch:
            datapoint = replay_buffer_pb2.MiniBatchResponse.DataPoint()
            datapoint.observation.CopyFrom(tf.make_tensor_proto(observation))
            datapoint.actions.extend([action.index for action in actions])
            for value, reward, policy in targets:
                target = replay_buffer_pb2.MiniBatchResponse.DataPoint.DataPointTarget(value=value, reward=reward)
                target.policy.probabilities.extend(policy)
                datapoint.targets.append(target)
            response.datapoints.append(datapoint)
        return response

    def Stats(self, request, context):
        return replay_buffer_pb2.StatsResponse(metrics=self.replay_buffer.stats())


class RemoteReplayBuffer:
    """
    A remote replay buffer, behaves exactly like ReplayBuffer.
    """
    def __init__(self, ip_port):
        channel = grpc.insecure_channel(ip_port)
        self.remote_replay_buffer = replay_buffer_pb2_grpc.ReplayBufferStub(channel)

    def save_history(self, game_history):
        request = replay_buffer_pb2.GameHistory()
        request.observations.extend([tf.make_tensor_proto(observation) for observation in game_history.observations])
        request.actions.extend([action.index for action in game_history.actions])
        request.rewards.extend(game_history.rewards)
        request.to_plays.extend([player.player_id for player in game_history.to_plays])
        request.root_values.extend(game_history.root_values)
        for probabilities in game_history.policies:
            policy = replay_buffer_pb2.Policy()
            policy.probabilities.extend(probabilities)
            request.policies.append(policy)
        response = self.remote_replay_buffer.SaveGame(request)
        return response.success

    def sample_batch(self, batch_size, num_unroll_steps, td_steps, discount):
        request = replay_buffer_pb2.MiniBatchRequest(batch_size=batch_size, num_unroll_steps=num_unroll_steps,
                                                     td_steps=td_steps, discount=discount)
        response = self.remote_replay_buffer.SampleBatch(request)

        batch = []
        for datapoint in response.datapoints:
            observation = tf.make_ndarray(datapoint.observation)
            actions = [Action(index) for index in datapoint.actions]
            targets = [(target.value, target.reward, target.policy.probabilities) for target in datapoint.targets]
            batch.append((observation, actions, targets))
        return batch

    def stats(self):
        return self.remote_replay_buffer.Stats(replay_buffer_pb2.Empty()).metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuZero Replay Buffer Server')
    parser.add_argument('--game', type=str, help='One of the games implemented in the games/ directory', required=True)
    parser.add_argument('--port', type=str, help='Port for gRPC communication', required=True)
    args = parser.parse_args()

    def serve(replay_buffer_config, port):
        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        servicer = ReplayBufferServer(replay_buffer_config)
        replay_buffer_pb2_grpc.add_ReplayBufferServicer_to_server(servicer, grpc_server)
        grpc_server.add_insecure_port('[::]:{}'.format(port))
        print('Starting replay buffer server, listening on port {}...'.format(port))
        grpc_server.start()
        grpc_server.wait_for_termination()

    config = load_game(args.game, parser)

    logging.basicConfig()
    serve(config.replay_buffer_config, args.port)
