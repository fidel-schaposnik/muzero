# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from muzero.protos import replay_buffer_pb2 as muzero_dot_protos_dot_replay__buffer__pb2


class ReplayBufferStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.NumGames = channel.unary_unary(
                '/tensorflow.muprover.ReplayBuffer/NumGames',
                request_serializer=muzero_dot_protos_dot_replay__buffer__pb2.Empty.SerializeToString,
                response_deserializer=muzero_dot_protos_dot_replay__buffer__pb2.NumGamesResponse.FromString,
                )
        self.SaveHistory = channel.unary_unary(
                '/tensorflow.muprover.ReplayBuffer/SaveHistory',
                request_serializer=muzero_dot_protos_dot_replay__buffer__pb2.GameHistory.SerializeToString,
                response_deserializer=muzero_dot_protos_dot_replay__buffer__pb2.NumGamesResponse.FromString,
                )
        self.SaveMultipleHistory = channel.stream_unary(
                '/tensorflow.muprover.ReplayBuffer/SaveMultipleHistory',
                request_serializer=muzero_dot_protos_dot_replay__buffer__pb2.GameHistory.SerializeToString,
                response_deserializer=muzero_dot_protos_dot_replay__buffer__pb2.NumGamesResponse.FromString,
                )
        self.SampleBatch = channel.unary_stream(
                '/tensorflow.muprover.ReplayBuffer/SampleBatch',
                request_serializer=muzero_dot_protos_dot_replay__buffer__pb2.MiniBatchRequest.SerializeToString,
                response_deserializer=muzero_dot_protos_dot_replay__buffer__pb2.MiniBatchResponse.FromString,
                )
        self.Stats = channel.unary_unary(
                '/tensorflow.muprover.ReplayBuffer/Stats',
                request_serializer=muzero_dot_protos_dot_replay__buffer__pb2.StatsRequest.SerializeToString,
                response_deserializer=muzero_dot_protos_dot_replay__buffer__pb2.StatsResponse.FromString,
                )
        self.BackupBuffer = channel.unary_stream(
                '/tensorflow.muprover.ReplayBuffer/BackupBuffer',
                request_serializer=muzero_dot_protos_dot_replay__buffer__pb2.Empty.SerializeToString,
                response_deserializer=muzero_dot_protos_dot_replay__buffer__pb2.GameHistory.FromString,
                )


class ReplayBufferServicer(object):
    """Missing associated documentation comment in .proto file."""

    def NumGames(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SaveHistory(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SaveMultipleHistory(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SampleBatch(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Stats(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def BackupBuffer(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ReplayBufferServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'NumGames': grpc.unary_unary_rpc_method_handler(
                    servicer.NumGames,
                    request_deserializer=muzero_dot_protos_dot_replay__buffer__pb2.Empty.FromString,
                    response_serializer=muzero_dot_protos_dot_replay__buffer__pb2.NumGamesResponse.SerializeToString,
            ),
            'SaveHistory': grpc.unary_unary_rpc_method_handler(
                    servicer.SaveHistory,
                    request_deserializer=muzero_dot_protos_dot_replay__buffer__pb2.GameHistory.FromString,
                    response_serializer=muzero_dot_protos_dot_replay__buffer__pb2.NumGamesResponse.SerializeToString,
            ),
            'SaveMultipleHistory': grpc.stream_unary_rpc_method_handler(
                    servicer.SaveMultipleHistory,
                    request_deserializer=muzero_dot_protos_dot_replay__buffer__pb2.GameHistory.FromString,
                    response_serializer=muzero_dot_protos_dot_replay__buffer__pb2.NumGamesResponse.SerializeToString,
            ),
            'SampleBatch': grpc.unary_stream_rpc_method_handler(
                    servicer.SampleBatch,
                    request_deserializer=muzero_dot_protos_dot_replay__buffer__pb2.MiniBatchRequest.FromString,
                    response_serializer=muzero_dot_protos_dot_replay__buffer__pb2.MiniBatchResponse.SerializeToString,
            ),
            'Stats': grpc.unary_unary_rpc_method_handler(
                    servicer.Stats,
                    request_deserializer=muzero_dot_protos_dot_replay__buffer__pb2.StatsRequest.FromString,
                    response_serializer=muzero_dot_protos_dot_replay__buffer__pb2.StatsResponse.SerializeToString,
            ),
            'BackupBuffer': grpc.unary_stream_rpc_method_handler(
                    servicer.BackupBuffer,
                    request_deserializer=muzero_dot_protos_dot_replay__buffer__pb2.Empty.FromString,
                    response_serializer=muzero_dot_protos_dot_replay__buffer__pb2.GameHistory.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'tensorflow.muprover.ReplayBuffer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ReplayBuffer(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def NumGames(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorflow.muprover.ReplayBuffer/NumGames',
            muzero_dot_protos_dot_replay__buffer__pb2.Empty.SerializeToString,
            muzero_dot_protos_dot_replay__buffer__pb2.NumGamesResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SaveHistory(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorflow.muprover.ReplayBuffer/SaveHistory',
            muzero_dot_protos_dot_replay__buffer__pb2.GameHistory.SerializeToString,
            muzero_dot_protos_dot_replay__buffer__pb2.NumGamesResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SaveMultipleHistory(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/tensorflow.muprover.ReplayBuffer/SaveMultipleHistory',
            muzero_dot_protos_dot_replay__buffer__pb2.GameHistory.SerializeToString,
            muzero_dot_protos_dot_replay__buffer__pb2.NumGamesResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SampleBatch(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/tensorflow.muprover.ReplayBuffer/SampleBatch',
            muzero_dot_protos_dot_replay__buffer__pb2.MiniBatchRequest.SerializeToString,
            muzero_dot_protos_dot_replay__buffer__pb2.MiniBatchResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Stats(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorflow.muprover.ReplayBuffer/Stats',
            muzero_dot_protos_dot_replay__buffer__pb2.StatsRequest.SerializeToString,
            muzero_dot_protos_dot_replay__buffer__pb2.StatsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def BackupBuffer(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/tensorflow.muprover.ReplayBuffer/BackupBuffer',
            muzero_dot_protos_dot_replay__buffer__pb2.Empty.SerializeToString,
            muzero_dot_protos_dot_replay__buffer__pb2.GameHistory.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
