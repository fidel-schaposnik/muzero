# MuProver

Tensorflow implementation of the MuZero algorithm, based on the pseudo-code provided
in the original paper:

**[1]** J. Schrittwieser, I. Antonoglou, T. Hubert, K. Simonyan, L. Sifre, S. Schmitt, A. Guez, 
E Lockhart, D. Hassabis, T. Graepel, T. Lillicrap, D. Silver,
["Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"](https://arxiv.org/abs/1911.08265)

## Design

This implementation isolates the various components of MuZero, and uses
 gRPC for communication between them. This should make it straightforward
to deploy the algorithm in the cloud and scale the resources up to the point
required for solving complex problems.

The main components are:

- An environment server (`environment`).

- A replay buffer server (`replay`), storing the self-played games and producing training 
batches from these.

- A network server (`network`), performing the neural network evaluations required during
self-play (provided by TensorFlow Serving).

- A training agent (`training`), using the self-played games from `replay` to train the 
  neural networks in `network`.

- A Monte-Carlo Tree-Search agent (`agent`), playing games using the latest networks 
available in `network` to produce games for `replay`.

## Installation

### Requirements

#### Install nvidia drivers for GPU support (optional)

Notice that we assume that system-wide nvidia drivers are installed. Installation of nvidia drivers is beyond the scope of this note. However, for Ubuntu 20.04 LTS and recent nvidia GPU's you can try
 ```
 sudo add-apt-repository ppa:graphics-drivers
sudo apt install ubuntu-drivers-common
ubuntu-drivers devices
sudo apt-get install nvidia-driver-450
```

#### Install TensorFlow Serving

Follow the instructions in https://www.tensorflow.org/tfx/serving/setup to install 
TensorFlow Serving. In short, add the TensorFlow Serving distribution URI as a
package source and then

```sudo apt-get update && sudo apt-get install tensorflow-model-server```

Alternatively, you can also run TensorFlow Serving in a Docker image (instructions 
at https://www.tensorflow.org/tfx/serving/docker ).

#### Install `conda`, `sshfs` and `screen`

### Installing muzero

Clone this git repository and install required dependencies 
(**TODO: streamline installation**).
 
#### Compiling protocol buffer files (optional)

You can (re)compile the protocol buffer files in the `protos` folder to generate the required
gRPC code:
```
python -m grpc_tools.protoc -I . -I PATH_TO_TENSORFLOW --python_out=. --grpc_python_out=. muzero/protos/environment.proto
python -m grpc_tools.protoc -I . -I PATH_TO_TENSORFLOW --python_out=. --grpc_python_out=. muzero/protos/replay_buffer.proto
```
Here `PATH_TO_TENSORFLOW` is the path to the tensorflow source code root folder,
containing `tensorflow/core/framework/tensor.proto` (you may clone it from 
https://github.com/tensorflow/tensorflow ).

#### Configuring the Tensorflow Serving `models.config` file

The file `models/models.config` specifies which models the TensorFlow Serving server 
will serve. In our case, this amounts to two separate models: `initial_inference` 
(combining `representation` and `prediction`) and `recurrent_inference` (combining 
`dynamics` and `prediction`). Each of these models has a `base_path` under which successive 
versions will be saved in separate directories. These should be absolute paths, so 
you should edit the `models/models.config` file accordingly (_e.g._ replace every 
occurrence of `%DIRECTORY%` in that file for whatever the output of 
`echo $PWD/models` is). This should not be necessary if you launch MuProver through 
the `./muprover.sh` script

**NOTE:** When using Docker images the `models` directory is mounted on the filesystem
root, so that `models/models.config` should simply point to `/models/initial_inference`
and `/models/recurrent_inference`, as shown in the file `models/docker_models.config`.

## Usage

Follow these steps to train MuZero to play a given game:

1. Start an environment server `environment` using

   ```
   python environment_services.py --game GAME --port PORT
   ```

   where `GAME` is one of the games implemented in the `games` directory 
   and `PORT` is the port for gRPC communication, _e.g._ 50000.
   
1. Start a replay buffer server `replay` using

   ```
   python replay_buffer_services.py --game GAME --port PORT --logdir LOG_DIR
   ```

   where `GAME` is one of the games implemented in the `games` directory 
   and `PORT` is the port for gRPC communication, _e.g._ 50001.
   
1. Start the training agent `training` using

   ```
   python training_services.py --game GAME --replay_buffer REPLAY_IP:PORT --min_games MIN_GAMES --saved_models MODELS_DIR --logdir LOG_DIR
   ```

   where `GAME` is one of the games implemented in the `games` directory,
   `REPLAY_IP:PORT` points to the replay buffer server of step 2 (_e.g._ 
   `localhost:50001`), and `MIN_GAMES` is the minimum number of games in the replay 
   buffer before training starts. The `--saved_models` argument should point to the 
   `MODELS_DIR` where the TensorFlow Serving server in step 4 will find its models 
   (this should be specified in the `models/models.config` file). The optional 
   `--logdir` argument results in exporting training statistics in TensorBoard 
   format to the `LOG_DIR` directory (as well as training checkpoints). You can find out about other optional arguments
   using `python training_services.py --help`.

1. Start the TensorFlow Serving neural network server `network` using

   ```
   tensorflow_model_server --port=PORT --rest_api_port=HTTP_PORT --model_config_file=models/models.config --enable_batching --batching_parameters_file=models/batching.config --monitoring_config_file=models/monitoring.config --file_system_poll_wait_seconds=15
   ```

   where `PORT` is the port for gRPC communication, _e.g._ 50002, and `HTTP_PORT`
   is the port for HTTP communication, _e.g._ 50003 (this can be used for testing 
   purposes, to see information about the networks or to obtain tensorflow-serving
   metrics).

   Alternatively, if using a Docker container the corresponding command is

   ```
   docker run -t --rm -p PORT:8500 -p HTTP_PORT:8501 --mount type=bind,source=$PWD/models,target=/models --name muzero_tfserver tensorflow/serving --model_config_file=/models/docker_models.config --enable_batching --batching_parameters_file=/models/batching.config --monitoring_config_file=/models/monitoring.config --file_system_poll_wait_seconds=15
   ```

   **NOTE:** If your system supports it, you can use the GPU-enabled docker container 
   by replacing the image name by `tensorflow/serving:latest-gpu` and including the 
   `--gpus=all` option.

1. Start one or more self-playing agents `agent` using

   ```
   python agent_services.py --game GAME --environment ENVIRONMENT_IP:PORT --replay_buffer REPLAY_IP:PORT --network NETWORK_IP:PORT --num_games NUM_GAMES
   ```
   
   where `GAME` is one of the games implemented in the `games` directory, the 
   `IP:PORT` pairs point to the servers of steps 1-3 (_e.g._ `localhost:50000`, 
   `localhost:50001` and `localhost:50002` respectively) and the optional 
   `--num_games` argument establishes the number of games the agent should play 
   (defaults to infinity if omitted).

#### Monitoring

- You can monitor the training progress using tensorboard by running 
`tensorboard --logdir LOG_DIR`.

- The TensorFlow Serving server exposes [Prometheus](http://prometheus.io/) metrics 
through HTTP at port `HTTP_PORT` defined in step 3 (e.g. http://localhost:50003/metrics).

#### Using `muprover.sh` to launch everything at once

A (very rough) bash script `muprover.sh` is provided to launch all the MuProver
processes at once on Linux systems. Invoke this script with the following syntax:

```
./muprover.sh -g GAME -c CONFIG_FILE -r MUPROVER_DIR -m MODELS_DIR -n RUN_NAME
```

where:
- `GAME` is one of the games implemented in the `games` folder.
- `CONFIG_FILE` is a configuration file following the structure described below.
- `MUPROVER_DIR` is the location (relative to `$HOME`) where the muprover code resides.
- `MODELS_DIR` is a directory containing the `models.config`, `batching.config` 
and `monitoring.config` files for the TensorFlow Serving server (typically the 
`models` directory in this repository).
- `RUN_NAME` is a unique name to assign to this run.

The configuration file is a series of lines of the form `service host:number`, where
`service` is one of `environment` (for the environment server), `replay` (for the replay
buffer server), `network` (for the TensorFlow Serving server), `training` (for the 
training service) and `agent` (for the self-playing agents). The `host` indicates where 
each service will be run, and the corresponding `number` is either the port for this
service (for `environment`, `replay` and `network`), the minimum number of games before
starting to train the networks (for `training`), or the number of agents to start (for 
`agent`). A sample configuration file is generated in `config.local`.

**NOTE:**

- Each of the `environment`, `replay`, `network` and `training` services should appear
exactly once in the configuration file, but there can be multiple `agent` lines.
- The hosts can be specified by IP addresses or domains, possibly prefixed by a `user@`;
use `localhost` to run a service locally
- The script assumes that all USER names and their HOME dirs are the same
- All communications occur through `ssh`, and we assume the current user has ~/.ssh/id_rsa.pub keys distributed
to ~/.ssh/authorized_keys to target hosts
- The script assumes in each host there is a `$HOME/MUPROVER_DIR` directory in which muzero python package
is installed under the  virtual environment `$HOME/MUPROVER_DIR/venv`
- The script assumes `screen` is present in all the hosts, and uses it to be able to 
monitor the various processes after they are launched.
- If the `training` and `network` services are in different hosts, the networks are saved
on the `network` host and the `training` host uses `sshfs` to save network snapshots 
there during training.

## Currently implemented games:

The following games have already been implemented (though only partial experiments 
have been carried out with them):

- [CartPole](https://github.com/openai/gym/wiki/CartPole-v0) (`games/cartpole.py`).
- TicTacToe (`games/random_tictactoe.py`).

#### Implementing other games

To implement a new game, you should sub-class the `Environment` class 
defined in `environment.py`, see `games/random_tictactoe.py` for an example. In 
the `games/yourgame.py` file you should also sub-class the `Network` class 
defined in `network.py` to define the neural networks used by MuProver for
your game. Finally, you should also provide a `make_config` method returning a
`MuZeroConfig` object (defined in `config.py`), containing all the
configuration parameters required by MuProver.

Alternatively, you may altogether skip creating an `Environment` sub-class
and simply define an environment server communicating through gRPC following
`protos/environment.proto`. If you do create the `Environment` sub-class, however, 
you will immediately be able to serve your environment using the standard server 
in `environment_services.py`.

## Custom training loops

You can define a custom training loop _e.g._ for synchronous training, 
whereby the same process alternates between self-playing games and training 
the neural networks. To do this, you may simply use the `Environment`,
`ReplayBuffer` and `Network` classes directly, instead of through their
`RemoteEnvironment`, `RemoteReplayBuffer` and `RemoteNetwork` counterparts.

However, you should be aware that this is certainly going to be much slower
than using the distributed, asynchroneous training.

## Notes

- You may want to tinker with `models/batching.config` and/or manually compile the 
TensorFlow Serving server to optimize network throughput in the target system.
