import grpc
import tensorflow as tf
from concurrent import futures

from muzero.protos import replay_buffer_pb2
from muzero.protos import replay_buffer_pb2_grpc

# For type annotations
from typing import Iterable, Tuple, Dict, Any

from muzero.utils import CommandLineParser

from muzero.replay_buffer import ReplayBuffer
from muzero.game_services import history_to_protobuf, history_from_protobuf
from muzero.exceptions import MuProverError

from muzero.config import MuZeroConfig
from muzero.game import GameHistory
from muzero.muprover_types import ObservationBatch, ActionBatch, ValueBatch, PolicyBatch


class ReplayBufferServer(replay_buffer_pb2_grpc.ReplayBufferServicer):
    """
    A server for replay buffers, exposing their functionality through a gRPC API.
    """
    def __init__(self, config: MuZeroConfig) -> None:
        self.replay_buffer = ReplayBuffer(config=config)

    def NumGames(self, request: replay_buffer_pb2.Empty, context) -> replay_buffer_pb2.NumGamesResponse:
        return replay_buffer_pb2.NumGamesResponse(num_games=self.replay_buffer.num_games())

    def SaveHistory(self, request: replay_buffer_pb2.GameHistory, context) -> replay_buffer_pb2.NumGamesResponse:
        self.replay_buffer.save_history(history_from_protobuf(request))

        return replay_buffer_pb2.NumGamesResponse(num_games=1)

    def SaveMultipleHistory(self,
                            request_iterator: Iterable[replay_buffer_pb2.GameHistory],
                            context
                            ) -> replay_buffer_pb2.NumGamesResponse:

        num_games = 0
        for message in request_iterator:
            self.replay_buffer.save_history(history_from_protobuf(message))
            num_games += 1
        return replay_buffer_pb2.NumGamesResponse(num_games=num_games)

    def SampleBatch(self,
                    request: replay_buffer_pb2.MiniBatchRequest,
                    context
                    ) -> Iterable[replay_buffer_pb2.MiniBatchResponse]:

        dataset = self.replay_buffer.as_dataset(batch_size=request.batch_size)

        for inputs, outputs in dataset:
            (batch_observations, batch_actions) = inputs
            (batch_target_rewards, batch_target_values, batch_target_policies) = outputs
            response = replay_buffer_pb2.MiniBatchResponse()
            response.batch_observations.CopyFrom(tf.make_tensor_proto(batch_observations))
            response.batch_actions.CopyFrom(tf.make_tensor_proto(batch_actions))
            response.batch_target_rewards.CopyFrom(tf.make_tensor_proto(batch_target_rewards))
            response.batch_target_values.CopyFrom(tf.make_tensor_proto(batch_target_values))
            response.batch_target_policies.CopyFrom(tf.make_tensor_proto(batch_target_policies))
            yield response

    def Stats(self, request: replay_buffer_pb2.StatsRequest, context) -> replay_buffer_pb2.StatsResponse:
        if request.detailed:
            return replay_buffer_pb2.StatsResponse(metrics=self.replay_buffer.detailed_stats())
        else:
            return replay_buffer_pb2.StatsResponse(metrics=self.replay_buffer.stats())

    def BackupBuffer(self, request: replay_buffer_pb2.Empty, context) -> Iterable[replay_buffer_pb2.GameHistory]:
        for history in self.replay_buffer.buffer:
            yield history_to_protobuf(history)


class RemoteReplayBuffer:
    """
    Connects to a replay buffer server and interacts with it.
    Behaves exactly like ReplayBuffer, but is agnostic about how the server deals with the actual buffer.
    """
    def __init__(self, ip_port: str) -> None:
        channel = grpc.insecure_channel(ip_port)
        self.remote_replay_buffer = replay_buffer_pb2_grpc.ReplayBufferStub(channel)

    def num_games(self) -> int:
        response = self.remote_replay_buffer.NumGames(replay_buffer_pb2.Empty())
        return response.num_games

    def save_games(self, filepath: str) -> None:
        response_iterator = self.remote_replay_buffer.BackupBuffer(replay_buffer_pb2.Empty())
        message = replay_buffer_pb2.GameHistoryList()
        message.histories.extend(response_iterator)

        with open(filepath, 'wb') as protobuf_file:
            protobuf_file.write(message.SerializeToString())

    def load_games(self, filepath: str) -> None:
        message = replay_buffer_pb2.GameHistoryList()
        with open(filepath, 'rb') as buffer_file:
            message.ParseFromString(buffer_file.read())
        self.remote_replay_buffer.SaveMultipleHistory(iter(message.histories))

    def save_history(self, game_history: GameHistory) -> None:
        request = history_to_protobuf(game_history)
        response = self.remote_replay_buffer.SaveHistory(request)
        if not response.num_games:
            raise MuProverError(message='Could not save game history!')

    def as_dataset(self,
                   batch_size: int
                   ) -> Iterable[Tuple[Tuple[ObservationBatch, ActionBatch], Tuple[ValueBatch, ValueBatch, PolicyBatch]]]:
        request = replay_buffer_pb2.MiniBatchRequest(batch_size=batch_size)
        response_iterator = self.remote_replay_buffer.SampleBatch(request)

        for response in response_iterator:
            batch_observations = ObservationBatch(tf.constant(tf.make_ndarray(response.batch_observations)))
            batch_actions = ActionBatch(tf.constant(tf.make_ndarray(response.batch_actions)))
            inputs = (batch_observations, batch_actions)

            batch_target_rewards = ValueBatch(tf.constant(tf.make_ndarray(response.batch_target_rewards)))
            batch_target_values = ValueBatch(tf.constant(tf.make_ndarray(response.batch_target_values)))
            batch_target_policies = PolicyBatch(tf.constant(tf.make_ndarray(response.batch_target_policies)))
            outputs = (batch_target_rewards, batch_target_values, batch_target_policies)
            yield inputs, outputs

    def stats(self) -> Dict[str, Any]:
        return self.remote_replay_buffer.Stats(replay_buffer_pb2.StatsRequest(detailed=False)).metrics

    def detailed_stats(self) -> Dict[str, Any]:
        return self.remote_replay_buffer.Stats(replay_buffer_pb2.StatsRequest(detailed=True)).metrics


def main():
    parser = CommandLineParser(name='MuProver Replay Buffer Server', game=True, port=True, threads=True)
    # parser.add_argument('--backup_dir', type=str, metavar='PATH',
    #                     help='Directory where game backups are stored')
    # parser.add_argument('--load', type=str, metavar='PATH',
    #                     help='Filename for .pbuf with games to load at startup.')
    args = parser.parse_args()

    # if args.backup_dir and not os.path.isdir(args.backup_dir):
    #     parser.error('--backup_dir {} does not point to a valid directory!'.format(args.backup_dir))

    # if args.load and not os.path.isfile(args.load):
    #     parser.error(f'--load {args.load} does not point to a valid .pbuf file!')

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.threads))
    servicer = ReplayBufferServer(config=args.config)
    replay_buffer_pb2_grpc.add_ReplayBufferServicer_to_server(servicer, grpc_server)
    grpc_server.add_insecure_port(f'[::]:{args.port}')
    print(f'Starting replay buffer server, listening on port {args.port}...')
    grpc_server.start()
    grpc_server.wait_for_termination()


if __name__ == '__main__':
    main()
