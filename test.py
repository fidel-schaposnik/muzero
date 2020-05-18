import logging
import grpc

from protos import replay_buffer_pb2_grpc
from protos import replay_buffer_pb2

class TestClient():
    def __init__(self, ip_port):
        channel = grpc.insecure_channel(ip_port)
        self.replay_buffer = replay_buffer_pb2_grpc.ReplayBufferStub(channel)

    def get_batch(self):
        request = replay_buffer_pb2.MiniBatchRequest(batch_size=2, num_unroll_steps=5, td_steps=10, discount=1.0)
        return self.replay_buffer.SampleBatch(request)


def run():
    client = TestClient('localhost:50001')
    batch = client.get_batch()
    print(batch)


if __name__ == '__main__':
    # logging.basicConfig()
    # run()
    # from games.cartpole import *
    # config = make_config()
    #
    # network = config.network_config.make_uniform_network()
    # network.save_model(r'C:\Users\fidel\Desktop\MuZeroData\models')