# MuZero

Tensorflow implementation of the MuZero algorithm, based on the pseudo-code provided
in the original paper:

**[1]** J. Schrittwieser, I. Antonoglou, T. Hubert, K. Simonyan, L. Sifre, S. Schmitt, A. Guez, 
E Lockhart, D. Hassabis, T. Graepel, T. Lillicrap, D. Silver,
["Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"](https://arxiv.org/abs/1911.08265)

**WARNING:** This code is highly experimental, badly documented and certainly buggy.
Comments, questions and corrections are welcome.

## Design

This implementation isolates the various components of MuZero, and uses
 gRPC for communication between them. This should make it straightforward
to deploy the algorithm in the cloud and scale the resources up to the point
required for solving more complex games.

The main components are:

- An environment server.

- A replay buffer server, storing the self-played games and producing from them
the training batches.

- A network server, performing the neural network evaluations required during
self-play (provided by `tensorflow-serving`).

- A Monte-Carlo Tree-Search agent, playing games using the latest networks available from a network server.

- A training agent, using the self-played games to train the neural networks and improve
gameplay.

## Usage

Follow these steps to train MuZero to play a given game:

1. Start an environment server using
`python environment.py --game GAME --port PORT`,
where `GAME` is one of the games implemented in the `games` directory 
and `PORT` is the port for gRPC communication, e.g. 50000.

1. Start a replay buffer server using
`python replay_buffer.py --game GAME --port PORT`,
where `GAME` is one of the games implemented in the `games` directory 
and `PORT` is the port for gRPC communication, e.g. 50001.
 
1. Start the `tensorflow-serving` neural network server in a Docker container using
`docker run -t --rm -p PORT:8500 -p HTTP_PORT:8501 --mount type=bind,source=$PWD/models,target=/models --name muzero_tfserver tensorflow/serving --model_config_file=/models/models.config --enable_batching --batching_parameters_file=/models/batching.config --monitoring_config_file=/models/monitoring.config`,
where `PORT` is the port for gRPC communication, e.g. 50002, and `HTTP_PORT`
is the port for HTTP communication (to see information about the networks or 
obtain tensorflow-serving metrics).

1. Start the training agent using
`python training.py --game GAME --replay_buffer REPLAY_IP:PORT --min_games MIN_GAMES --saved_models_path $PWD\models --logging_path $PWD\logs`,
where `GAME` is one of the games implemented in the `games` directory,
`REPLAY_IP:PORT` point to the replay buffer server of step 2, and `MIN_GAMES`
is the minimum number of games in the replay buffer before training starts. 

1. Start one or more self-playing agents using
`python agent.py --game GAME --environment ENVIRONMENT_IP:PORT --replay_buffer REPLAY_IP:PORT --network NETWORK_IP:PORT --num_games NUM_GAMES`,
where the `IP:PORT` pairs point to the servers of steps 1-3.

#### Monitoring

- You con monitor the training progress using tensorboard by running `tensorboard --logdir $PWD\logs`.

- The tensorflow-serving server exposes [Prometheus](http://prometheus.io/) metrics through HTTP at port `HTTP_PORT`
defined in step 3 (e.g. http://localhost:50003/metrics).

- You may monitor the replay buffer and networks using `python monitor.py --replay_buffer REPLAY_IP:PORT`,
which starts a Flask server on port 5000 showing various statistics.

## Currently implemented games:

The following games have already been implemented (though only partial experiments 
have been carried out with them):

- [CartPole](https://github.com/openai/gym/wiki/CartPole-v0) (`games/cartpole.py`)

#### Implementing other games

To implement a new game, you should sub-class the `Environment` class 
defined in `environment.py`, see `games/cartpole.py` for an example. In 
the `games/yourgame.py` file you should also sub-class the `Network` class 
defined in `network.py` to define the neural networks used by MuZero for
your game. Finally, you should also provide a `make_config` method returning a
`MuZeroConfig` object (defined in `config.py`), containing all the
configuration parameters required by MuZero.

Alternatively, you may altogether skip creating an `Environment` sub-class
and simply define an environment server communicating through gRPC following
`protos/environment.proto`.

## Custom training loops

You can define a custom training loop e.g. for synchroneous training, 
whereby the same process alternates between self-playing games and training 
the neural networks. To do this, you may simply use the `Environment`,
`ReplayBuffer` and `Network` classes directly, instead of through their
`RemoteEnvironment`, `RemoteReplayBuffer` and `RemoteNetwork` counterparts.   

However, you should be aware that this is certainly going to be much slower
than using the distributed, asynchroneous training.

## Notes

- You may want to tinker with `models/batching.config` and/or use a different
tensorflow-serving Docker image to optimize network throughput in your system.
