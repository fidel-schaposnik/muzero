import tensorflow as tf

from protos import replay_buffer_pb2
from game import GameHistory
from utils import to_bytes_dict, from_bytes_dict

# For type annotations
from typing import List

from muzero_types import Observation, Policy, Value, Action, Player


def history_to_protobuf(game_history: GameHistory) -> replay_buffer_pb2.GameHistory:
    message = replay_buffer_pb2.GameHistory()
    message.observations.extend([tf.make_tensor_proto(obs) for obs in game_history.observations])
    message.to_plays.extend(game_history.to_plays)
    message.actions.extend(game_history.actions)
    message.rewards.extend(game_history.rewards)
    message.root_values.extend(game_history.root_values)
    message.policies.extend([tf.make_tensor_proto(policy) for policy in game_history.policies])
    message.metadata.update(to_bytes_dict(game_history.metadata))
    return message


def history_from_protobuf(message: replay_buffer_pb2.GameHistory) -> GameHistory:
    game_history = GameHistory()
    game_history.observations = [Observation(tf.constant(tf.make_ndarray(obs))) for obs in message.observations]
    game_history.to_plays = [Player(id) for id in message.to_plays]
    game_history.actions = [Action(index) for index in message.actions]
    game_history.rewards = [Value(reward) for reward in message.rewards]
    game_history.root_values = [Value(root_value) for root_value in message.root_values]
    game_history.policies = [Policy(tf.constant(tf.make_ndarray(policy))) for policy in message.policies]
    game_history.metadata.update(from_bytes_dict(message.metadata))
    return game_history


def save_games(game_histories: List[GameHistory], filepath: str) -> None:
    message = replay_buffer_pb2.GameHistoryList()
    message.histories.extend([history_to_protobuf(history) for history in game_histories])

    with open(filepath, 'wb') as protobuf_file:
        protobuf_file.write(message.SerializeToString())
    print(f'Saved {len(game_histories)} games to {filepath}!')


def load_games(pbuf_filename: str) -> List[GameHistory]:
    message = replay_buffer_pb2.GameHistoryList()
    with open(pbuf_filename, 'rb') as buffer_file:
        message.ParseFromString(buffer_file.read())
    game_histories = [history_from_protobuf(history_pbuf) for history_pbuf in message.histories]
    print(f'Loaded {len(game_histories)} games!')

    return game_histories
