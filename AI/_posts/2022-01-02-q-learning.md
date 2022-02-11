---
layout: post
author: Simon
title: Noughts and Crosses with Q-learning
---

Reinforcement Learning is something I've been keen to get to understand a bit more for a while. I've always had a high-level understanding but never actually _done_ it. Ultimately, I'd like to play with deep Q-learning but that felt like a bit more than I wanted to bite off, so started with standard Q-learning.

A Jupyter notebook accompanying this post is available [here](https://github.com/sdrogers/ml_examples/blob/main/q_learning/Q%20learning%20for%20tic-tac-toe.ipynb)

## Q-learning

Q-learning is a popular method within the field of Reinforcement Learning (RL). The goal of RL is to train an _agent_ to navigate through a state space in an optimal way. As an example, I'm using the game of tic-tac-toe (aka noughts and crosses).

The _agent_ is one of the tic-tac-toe players. In everything below, it plays against another computer player although there's no reason it couldn't play against a human. The goal of RL is to train this _agent_ such that it can make sensible moves based on the current board situation (the _state_).

Whereas the standard ML paradigm is to train models from a static training set (i.e. collect training data -> train -> predict), in RL the training is done via the _agent_ actually performing the task, over and over (and over and over and over) again. The agent receives _rewards_ depending on how well it does, and modifies its behaviour such that actions that gave it high rewards are more likely to happen again.

At any point in a game of tic-tac-toe, the _state_ is determined by the current orientation of noughts and crosses on the board. The _agent_ can choose one of a fixed set of actions (where to place their marker). Q-learning attempts to learn a value for every pair of state and action. The higher the value, the better that action is from the current state.

To make this more concrete, let's use the numbers 1 to 9 to define the positions on the board:

1 2 3
4 5 6
7 8 9

And, assume that the current state of the board is (where 1 and 2 stand for players 1 and 2 respectively):

1 0 2
0 1 2
0 0 0

If our agent is player 1, then there are 5 possible actions (positions 2, 4, 7, 8, and 9). A trained Q-learner would have access to Q-values for each of these actions and might choose the move that had the highest Q-value. In this example, we'd hope that for this state, it would have the highest score for action 9, which is obviously the best move to make.

## The Q-table

The Q-learner keeps track of Q-values within a Q-table. As they have to store scores for all valid pairs of state and action, Q-tables can become enormous. In my Python example, I use a dictionary to store the Q-table, where the key is be a String representation of the current board (state). For example, the board state shown above could be flattened into a list:

```python
state = [1, 0, 2, 0, 1, 2, 0, 0, 0]
state_string = "102012000"
```

The Q-table dictionary would have one key, value pair for each possible state. In my implementation I encode the value in the dictionary as a list of Q-values for this state. E.g. for this state:

```python
q_table["102012000"] = [0, 1, 0, 0.5, 0, 0, 2.0, 0.5, 23.2]
```
Suggesting that action 9 (bottom right corner) is the best action. Note that in my implementation I always define the action list as having length 9 despite the fact that some moves are invalid. This doesn't really matter as long as we implement the agent to choose the _valid_ move with the highest Q-value and not just the move with the highest Q-value.

## Updating Q-values

At the heart of Q-learning is how we update the values in the Q-table. This is the _learning_ part and, in my implementation, is done at the end of every game the agent plays. Updating is done using Bellman's equation:

$$ Q_{new}(S_t, a_t) = (1-\alpha) Q(S_t, a_t) + \alpha(r_t + \lambda \max_a Q(S_{t+1}, a)) $$

Let's look at the various parts of this. Bellman's equation tells us how to update the Q-value of action $$a_t$$ from state $$S_t$$. Taking this action will result in moving to $$S_{t+1}$$. Taking action $$a_t$$ from state $$S_t$$ might  also result in the agent receiving a reward $$r_t$$ (if, e.g. the move ends the game).

$$\alpha$$ is known as the learning rate, and controls how much Q-values can change based on a single observation (usually $$\alpha$$ is pretty small). $$\lambda$$ is a parameter that controls how much future state Q-values influence the current one.

Piecing this together, the Q-value of $$a_t$$ (from state $$S_t$$) is updated by taking a weighted sum of its current value, and the combination of the reward obtained plus the best Q-value possible from the state reached by performing $$a_t$$.

In the tic-tac-toe example, when a game is finished, all of the Q-values corresponding to actions that the agent took are updated using this equation, starting with the final one and moving backwards. If the final move results in a win, a large positive value (I use 20) might be given as the reward. The Q-value corresponding to this action in the relevant state will therefore be updated from, say, $$g$$ to $$(1 - \alpha)g + 20\alpha$$. This will almost certainly mean it increases, and therefore makes the winning move more likely. We then move back to the move the agent made before the winning one. There is no reward for this move and, assuming that the updated value of $$g$$ is the highest Q-value for the state the agent reached, the Q-value will be updated from, e.g. $$h$$ to $$(1-\alpha)h + \alpha\lambda g$$. Again, this is likely to increase $$h$$ making the move more likely.

Conversely, if the final move lead to a defeat, the reward might be a large negative value (I use -20) with the result that all Q-values corresponding to the actions taken are likely to _decrease_ and become less likely in future.


## Limitations

A couple of limitations emerge from this description of Q-learning. Firstly, for many problems, the state space is enormous and the Q-table is going to become impractical to learn. Secondly, assuming that one needs to visit each state multiple times during training to obtain reliable Q-values suggests that training is going to take a _long_ time. Deep Q-learning attempts to overcome this: rather than keep a table of all Q-values, it trains a neural network to _predict_ Q-values for states -- more on this in another post once I've had a go with it.


## Exploration strategies

A fully trained agent could, in theory, always choose the action corresponding to the highest Q-value. However, during training this is a really bad idea as it can result in the agent getting stuck in a rut in which it keeps just repeating the same moves. An alternative is to add some randomness into the agents strategy during training. Epislon-greedy exploration is one such strategy in which the agent either takes the action with the highest Q-value, or chooses an action at random. The probability of doing the random choice is $$\epsilon$$ and often $$\epsilon$$ is decayed during training so that at the start the agent makes many random moves (_exploration_) before gradually making fewer and fewer (_exploitation_) until it only ever takes the best strategy. In my experiments, I keep $$\epislon$$ fixed for an initial number of games, before starting an exponential decay that leaves it at approximately zero at the end of the training phase.

## Experiments

All of the code for this is in a Jupyter notebook [here](https://github.com/sdrogers/ml_examples/blob/main/q_learning/Q%20learning%20for%20tic-tac-toe.ipynb).

Initially, I use the `random` player provided by `kaggle-environments`. This player picks a valid move at random. 5000 training games are played, and $$\epsilon$$ is fixed at 1 for the first 2000 and then decays exponentially to approximately 0 over the next 3000. The agent always plays as player 1 and always goes first. Every 100 training games, the agent plays 100 games against the opponent to assess what proportion it wins. The output of a game will be +1 if the agent wins, 0 if it is a draw, and -1 if it loses. Averaging this value gives a summary of performance between -1 and 1, the higher the better. This is how that performance changes as training progresses (the dashed line is the point at which $$\epsilon$$ begins to decay):

<img src='/assets/q_learn_random_performance.png'>

The agent learns pretty quickly to beat a random opponent: by the time it has played around 2000 training games, it wins almost all of the time. Note that we might imagine the performance would start at 0, but in fact it starts at about 0.2. This is probably just the advantage our agent gets for always going first. We can also plot how many states it has explored as a function of the number of training games:

<img src='/assets/q_learn_random_states.png'>

It's also interesting to look at values within the Q-table. Here's the value that has been learnt for the initial state:

```
2.8  2.2  2.8
2.2  9.4  2.2
2.8  2.2  2.8
```
which suggests that against the random opponent, starting in the center is the best thing to do.

## Leveraging Symmetry

For any state in the tic-tac-toe board, there are three other exactly equivalent states produced by rotating the board 90, 180 and 270 degrees. The code above makes use of this by simply running the Q-table updated four times after each game: once with what actually happened and then three other times for the different rotations of the whole game. It might be possible to do something smarter by, e.g. only storing one version of each position in the Q-table and then just rotating the actions, or converting each state into some kind of canonical representation. Both would take effort though, and given that the size of the Q-table in this example isn't prohibitve, replicating the games was an easier option.

## The Reaction opponent

`kaggle-environments` also has an opponent called `reaction`. This opponent will always form and stop lines when they can. The equivalent plots to those above for this opponent are:

<img src='/assets/q_learn_reaction_performance.png'>

and

<img src='/assets/q_learn_reaction_states.png'>

It's worth noting that in this case, the initial performance of the agent is much worse. This is to be expected as the opponent will now make moves that stop the agent winning, or cause the agent to lose when it is able to. That said, the agent still learns pretty quickly how to win.

In the random example, the optimal opening move was the centre. Interestingly, against `reaction` we get the following Q-values for the initial state:

```
6.2  0.5  6.2
0.5  1.1  0.5
6.2  0.5  6.2
```
The corners are now the optimal place to start. I guess this highlights a key point of Q-learning -- the learner will learn to beat the opponent that it trains against and therefore will not always converge on the same strategy.

## Towards deep Q-learning

A significant limitation of Q-learning is that the agent has no ability to _generalise_ between different states. In other words, it can only learn Q-values for the state $$S_t$$ by visiting $$S_t$$ (potentially multiple times) during training. It doesn't matter at all if it has good Q-values for some other state $$S_t'$$ which is almost exactly the same as $$S_t$$. In many real applications, it may not be possible to train for long enough to experience all states enough times to learn Q-values.

Deep Q-learning is an attempt to overcome this problem -- instead of learning a Q-table, it _approximates_ a Q-table with a predictive function. In particular it uses a neural network that is fed the state and predicts the Q-values of each action. The key benefit is that the neural network can _generalise_. Knowledge about good Q-values for state $$S_t$$ will help with predictions for $$S_t'$$.