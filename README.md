# MuZero

Tensorflow implementation of MuZero algorithm.

**Main differences with the algorithm described in https://arxiv.org/abs/1911.08265**

- Completely synchroneous: alternates game generation and network training.

- Allows for more flexibility in the environment responses: after each move all players
 can receive rewards (not just the player who made that move).
 
 - Includes an additional head in the dynamics function to predict who is the next player
  to play.

- Uses a simplified UCB formula

- Uses no value normalization

## Currently implemented games:

- Tic-tac-toe

## Other features

- Tensorboard logging