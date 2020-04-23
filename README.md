# MuZero

Tensorflow implementation of the MuZero algorithm, based on the pseudo-code provided
in the original paper:

**[1]** J. Schrittwieser, I. Antonoglou, T. Hubert, K. Simonyan, L. Sifre, S. Schmitt, A. Guez, 
E Lockhart, D. Hassabis, T. Graepel, T. Lillicrap, D. Silver,
["Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"](https://arxiv.org/abs/1911.08265)

**WARNING:** This code is highly experimental, badly documented and certainly buggy.
Comments, questions and corrections are welcome.

## Main differences with the algorithm described in [1]

Some changes have been made in the direction of supporting multi-player games:

- **More flexibility in the environment responses:** after each move all players can receive
  rewards (not just the player who made that move).
 
- **An additional head in the dynamics function** predicts who is the next player to play.

Additionally, a simplified UCB formula is used to reduce the number of hyperparameters.

## Training modes

Support for single-thread training (_synchronous mode_) and multi-thread or distributed 
training (_asynchronous mode_).

#### Synchronous mode

In this mode a single thread plays games to generate training data, and then uses this 
data to train the neural networks. This is slower but easier to setup than asynchronous 
training (as used in the original paper). Use

`python muzero.py --game GAME --synchronous --num-steps NUM_STEPS --num-games NUM_GAMES --num-eval-games NUM_EVAL_GAMES`

to alternate self-playing `NUM_GAMES` games, and training for `NUM_STEPS` steps. At each 
checkpoint, `NUM_EVAL_GAMES` are played to evaluate the network.

#### Asynchronous mode

In this mode self-playing and training occur simultaneously in different processes or 
different nodes of a distributed network. A simple HTTP server maintains a database of 
self-played games and neural network weights. Self-playing and training agents interact 
with this server through a simple API.

- Use  `python muzero.py --game GAME --server DATA_DIR` to start the server and save logs in 
 `DATA_DIR`(network weights are saved in [HDF5 format](http://www.h5py.org/), self-play 
 games are pickled). You can then go to http://localhost:5000/ to see basic server 
 statistics.

- Use `python muzero.py --client HOST --self-play NUM_GAMES` to start a self-playing agent
 that uses the latest network from the server `HOST` to generate batches of `NUM_GAMES` 
 games and send them back to the server.
 
- Use `python muzero.py --client HOST --train NUM_EVAL_GAMES` to start a training agent 
 that queries the server at `HOST` for batches of training data, and uses them to train 
 the latest network. At each checkpoint, `NUM_EVAL_GAMES` are played to evaluate the network.

By default, the server is only visible locally. Change `api.run()` to `api.run(0.0.0.0)` in
`storage_replay.py` to make the server visible to the outside

**WARNING:** the API is implemented using [Flask](https://flask.palletsprojects.com/), and it is not recommended to deploy it as is 
for production. 

## Currently implemented games:

The following games have already been implemented (though only partial experiments 
have been carried out with them):

- [CartPole-v0](https://github.com/openai/gym/wiki/CartPole-v0) (`games/cartpole.py`)
- Tic-tac-toe (`games/tictactoe.py`)
- One-arm bandit (`games/onearmbandit.py`)

You can run MuZero on any of these games by using the `--game` command-line 
argument with the corresponding filename, e.g. `--game tictactoe`.
 

## Other features

- **Tensorboard logging:** on the base directory use `tensorboard --logdir checkpoints` 
to visualize training.

- **Easily add games**: just add a file to the `games` directory defining MuZero's
configuration for the game of your choice, and implementing sub-classes for the 
`Environment`, `Game` and `Network` classes. The methods you need to implement 
for each subclass are marked in `environment.py`, `game.py` and `network.py`, 
respectively. **NOTE:** It should be trivial to interface with [OpenAI Gym](https://gym.openai.com/)
environments, see `games/cartpole.py` for an example.

- **Loss selection:** you can choose to use MSE or CCE losses for values and rewards
(setting `scalar_support_size` in the game configuration transforms scalars to categorical
representations in a manner similar to that described in [[1]](https://arxiv.org/abs/1911.08265)).

- **Weight and game buffer loading in asynchronous mode:** you can upload network weights
and self-played games directly to the server in asynchronous mode in order to resume 
training from a checkpoint of your choice.

- **Playing against the latest network:** a basic interface is setup through the 
server to play against the latest network directly on the browser in 
asynchronous mode.

- **Pre-built networks:** basic residual and fully connected architectures are defined
in `network.py`, and can be reused simply defining the network parameters 
(see examples in `games` directory).

## To-do list

- **Prioritized replay**
- **Optimize hyperparameters for some of the included weights**
- **Pre-trained networks**