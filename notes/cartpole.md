# CartPole notes

## Game version

The OpenAI Gym provides two versions of the CartPole environment:

- `CartPole-v0` has episode length 200 and is solved when the average
reward for 100 episodes is 195.

- `CartPole-v1` has episode length 500 and is solved when the average
reward for 100 episodes is 475.

## Architecture

Due to the nature of the observations, we replace MuZero's original residual
networks with fully connected networks.

We extend the environment's observation
space to include an "episode progress" dimension in the range [0,1). Otherwise,
MuZero has no way to know if it is close to episode termination, and therefore
fails to accurately predict the value of positions in later stages of the
episode.

## Training

##### Discount

The discount (<img src="https://render.githubusercontent.com/render/math?math=\gamma">)
introduces an effective horizon in the number of future moves MuZero learns from:

- If <img src="https://render.githubusercontent.com/render/math?math=\gamma \simeq 1">,
the horizon is far away and MuZero may use unreliable data for training 
(e.g. it may assume the game will go on after the episode terminates).

- If <img src="https://render.githubusercontent.com/render/math?math=\gamma \ll 1">,
the horizon is too close for MuZero to detect potentially bad choices (e.g. slowly drifting to one side).  

##### Losses

Value loss dominates the total loss, since values roughly scale with episode 
length whereas other losses remain O(1). Initially the network predicts 
values close to 0 and acts randomly. The typical length of episodes is then 
around L = 10 steps, and so with uniform sampling we have an initial loss of

<img src="https://render.githubusercontent.com/render/math?math=L_v = \frac{1 - 2\gamma - 2\gamma^2 %2B 2\gamma^{(2 %2B L)} %2B 2 \gamma^{(3 %2B L)} - \gamma^{(4 %2B 2 L)} %2B L (1 - \gamma^2)}{(1 %2B L) (1-\gamma)^3 (1 %2B \gamma)}"><br>

For <img src="https://render.githubusercontent.com/render/math?math=\gamma = 0.998"> 
this results in <img src="https://render.githubusercontent.com/render/math?math=L_v \simeq 45.3">,
whereas <img src="https://render.githubusercontent.com/render/math?math=L_r \simeq 1">
and <img src="https://render.githubusercontent.com/render/math?math=L_p \simeq \log2 = 0.6931\dots">,
so the total loss is <img src="https://render.githubusercontent.com/render/math?math=L \simeq 47">.

The fact that the value loss dominates makes it possible to find 
local minima where the dynamics network predicts rewards close to 0, yet 
MuZero has effectively learned to solve the environment. Eventually, the 
weights decay into better minima where rewards are accurately predicted 
(using an optimizer with some sort of momentum may be important for this).

##### Instabilities

As MuZero learns, episodes get longer so even if values are better approximated
the total loss may increase over time. If this is left unchecked, value loss may 
completely overwhelm all the other terms in the loss function, in which case 
MuZero can un-learn how to play. To see how this happens, consider the situation 
where MuZero has learned to accurately predict rewards, so <img src="https://render.githubusercontent.com/render/math?math=L_r \simeq 0">
but episodes are still typically short (i.e. almost random play). If suddenly 
typical episode length jumps to much higher values (i.e. MuZero learns to play 
much better), and we have a bootstrap length (`config.td_steps`) of, say, <img src="https://render.githubusercontent.com/render/math?math=T = 50">,
target values will increase by roughly
 
<img src="https://render.githubusercontent.com/render/math?math=\Delta V \approx \frac{\gamma^{T %2B 1} - 1}{\gamma - 1} \simeq 48.5"><br>

where we continue to use <img src="https://render.githubusercontent.com/render/math?math=\gamma = 0.998">
for illustration purposes. This is because MuZero will look T steps ahead 
of most positions before predicting values, whereas before it looked ahead only
a few steps (until episode termination). Thus, the average value loss would 
increase to about <img src="https://render.githubusercontent.com/render/math?math=L_v \simeq 2.4 \times 10^3">,
meaning that the reward loss <img src="https://render.githubusercontent.com/render/math?math=L_r \leq 1 \ll L_v">
stops effectively contributing to the gradients.
