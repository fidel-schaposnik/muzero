import grpc
import tensorflow as tf
from concurrent import futures
from threading import RLock

from muzero.environment import Environment
from muzero.exceptions import MuProverEnvironmentError
from muzero.utils import CommandLineParser, random_id, to_bytes_dict, from_bytes_dict
from muzero.protos import environment_pb2
from muzero.protos import environment_pb2_grpc

# For type annotations
from typing import Tuple, Callable, Dict, List

from muzero.muprover_types import State, Value, Action
from muzero.config import MuZeroConfig


class RemoteEnvironmentServer(environment_pb2_grpc.RemoteEnvironmentServicer):
    """
    A server for environments, exposing their functionality through a gRPC API.
    Implements the methods defined in the protos/environment.proto file.
    """
    def __init__(self, environment_class: Callable[..., Environment]) -> None:
        self.environment_class: Callable[..., Environment] = environment_class
        self.environments: Dict[str, Environment] = {}
        self.environments_lock: RLock = RLock()

    @staticmethod
    def _make_state(observation, legal_actions):
        state = environment_pb2.State()
        state.observation.CopyFrom(tf.make_tensor_proto(observation))
        state.legal_actions.extend(legal_actions)
        return state

    def _get_environment(self, environment_id: str):
        with self.environments_lock:
            return self.environments[environment_id] if environment_id in self.environments.keys() else None

    def Initialization(self,
                       request: environment_pb2.InitializationRequest,
                       context
                       ) -> environment_pb2.InitializationResponse:

        environment_parameters = from_bytes_dict(request.environment_parameters)
        environment_id = random_id()
        environment = self.environment_class(**environment_parameters)
        with self.environments_lock:
            self.environments[environment_id] = environment

        print(f'started environment with id={environment_id}, parameters={environment_parameters}')
        return environment_pb2.InitializationResponse(success=True, environment_id=environment_id)

    def Finalization(self,
                     request: environment_pb2.FinalizationRequest,
                     context
                     ) -> environment_pb2.FinalizationResponse:

        with self.environments_lock:
            try:
                del(self.environments[request.environment_id])
            except KeyError:
                return environment_pb2.FinalizationResponse(success=False)
            else:
                print(f'finalized environment with id={request.environment_id}')
                return environment_pb2.FinalizationResponse(success=True)

    def Step(self, request: environment_pb2.StepRequest, context) -> environment_pb2.StepResponse:
        environment = self._get_environment(request.environment_id)
        if environment is None:
            return environment_pb2.StepResponse(success=False)

        try:
            observation, reward, done, info = environment.step(Action(request.action))
        except MuProverEnvironmentError:
            return environment_pb2.StepResponse(success=False)
        else:
            return environment_pb2.StepResponse(success=True,
                                                state=self._make_state(observation, environment.legal_actions()),
                                                reward=reward,
                                                done=done,
                                                info=to_bytes_dict(info))

    def Reset(self, request: environment_pb2.ResetRequest, context) -> environment_pb2.ResetResponse:
        environment = self._get_environment(request.environment_id)
        if environment is None:
            return environment_pb2.ResetResponse(success=False)

        observation = environment.reset()
        return environment_pb2.ResetResponse(success=True,
                                             state=self._make_state(observation, environment.legal_actions()))


class RemoteEnvironment(Environment):
    """
    Connects to an environment server and interacts with it.
    Behaves exactly like Environment, but is agnostic about how the server deals with the environment.
    """
    def __init__(self, config: MuZeroConfig, ip_port: str) -> None:
        super().__init__(action_space_size=config.game_config.action_space_size)

        self.environment_parameters: Dict[str, bytes] = to_bytes_dict(config.game_config.environment_parameters)
        self.ip_port: str = ip_port
        print(f'game_config.environment_parameters: {config.game_config.environment_parameters}')

        self._legal_actions: List[Action] = []

    def __enter__(self) -> 'RemoteEnvironment':
        self._channel = grpc.insecure_channel(self.ip_port)
        self._environment_stub = environment_pb2_grpc.RemoteEnvironmentStub(self._channel)

        request = environment_pb2.InitializationRequest(environment_parameters=self.environment_parameters)
        response = self._environment_stub.Initialization(request)
        if not response.success:
            raise MuProverEnvironmentError(message='failed to initialize remote environment')

        self._environment_id = response.environment_id
        return self

    def __exit__(self, *args) -> bool:
        request = environment_pb2.FinalizationRequest(environment_id=self._environment_id)
        response = self._environment_stub.Finalization(request)
        if not response.success:
            raise MuProverEnvironmentError(message='failed to finalize remote environment')
        self._channel.close()
        # propagate exceptions
        return False

    def step(self, action: Action) -> Tuple[State, Value, bool, dict]:
        request = environment_pb2.StepRequest(environment_id=self._environment_id, action=action)
        response = self._environment_stub.Step(request)
        if not response.success:
            raise MuProverEnvironmentError(f'failed to perform action {action}')

        self._legal_actions = [Action(index) for index in response.state.legal_actions]
        info = from_bytes_dict(response.info)
        return tf.constant(tf.make_ndarray(response.state.observation)), Value(response.reward), response.done, info

    def reset(self) -> State:
        request = environment_pb2.ResetRequest(environment_id=self._environment_id)
        response = self._environment_stub.Reset(request)
        if not response.success:
            raise MuProverEnvironmentError(message='failed to reset the environment')

        self._legal_actions = [Action(index) for index in response.state.legal_actions]
        return tf.constant(tf.make_ndarray(response.state.observation))

    def is_legal_action(self, action: Action) -> bool:
        return action in self._legal_actions


def main():
    parser = CommandLineParser(name='MuProver Environment Server', game=True, port=True, threads=True)
    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.threads))
    servicer = RemoteEnvironmentServer(environment_class=args.config.game_config.environment_class)
    environment_pb2_grpc.add_RemoteEnvironmentServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{args.port}')
    print(f'Starting server for environment {args.config.game_config.name} at port {args.port}...')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
