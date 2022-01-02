---
layout: post
author: Simon
title: Reinforcement Learning (Q-learning)
---

Reinforcement Learning is not something I've really had any experience with. The following post is made up of some notes I made for myself whilst playing around with Q-learning with Python and `kaggle-environments` in an attempt to become more familiar with RL and Q-learning, in order to then move on to trying out deep Q-learning.

## Spaces, Agents, and Actions

Many useful tasks can be expressed using the language of _states_ and _agents_. An _agent_ exists in some world, the configuration of which at some time instant is known as its _state_. _Agents_ can perform _actions_ which might change the _state_, or move them to another _state_. In such tasks our goal will often be to come up with a strategy for the _agent_ that determines the optimal _action_ to take in any particular _state_ to help solve the task at hand.

To make this less abstract, consider the game of chess. The _agent_ is playing against some other player (could be a human, could be another artificial agent). At any point in the game, the _state_ is the current configuration of pieces on the board. The possible _actions_ are the moves that the _agent_ can make and the task is to try and win. Another example is a driverless car: at any time instant, the state is determined by the data coming from all of its sensors (cameras, microphones, etc), _actions_ would be steering, accelerating, brakeing, etc and the task might be to get from one location to another (safely).

## Reinforcement Learning

Reinforcement Learning (RL) is a family of techniques developed to train agents to perform well in tasks like this. Agents that initially behave randomly are allowed to attempt the task many times. Reward values are provided depending on how well they do, and these reward values are used to positively and negatively reinforce the actions that led to them. For example, in a game of chess, a high positive reward might be provided if the agent wins the match and the agent is updated to that the choices that led to the win are more likely in the future. If the agent loses, a large negative reward might be used to update the agent to make the moves that led to defeat less likely.

## Q-learning

Q-learning is a popular algorithm for RL. Q-learning ultimately ascribes a numerical value to each state, action pair. The higher the number, the better. I.e. the actions with the highest values from a particular state are those that are most likely to lead to success. These values are known as Q-values, and are optimised via RL during training. The values are stored in a Q-table. You can think of this as a table with one row per state, and one column per action. The values stored in the table are the Q-values. Armed with a particular Q-table, an agent that finds itself in a particular state can simply lookup the action that has the highest Q-value in the row corresponding to that state.

### Limitations

A couple of limitations emerge from this description of Q-learning. Firstly, for many problems, the state space is enormous and the Q-table is going to become really big. Secondly, assuming that one needs to visit each stage multiple times during training to obtain reliable Q-values suggests that training is going to take a _long_ time. Deep Q-learning attempts to overcome this: rather than keep a table of all Q-values, it trains a neural network to _predict_ Q-values for states -- more on this in another post once I've had a go with it.

## Updating Q-values

At the heart of Q-learning is how to update the values in the Q-table. This is done according to Bellman's equation:

$$ Q_{new}(S_t, a_t) = (1-\alpha) Q(S_t, a_t) + \alpha(r_t + \lambda \max_a Q(S_{t+1}, a)) $$

Let's look at the various parts of this. Bellman's equation is for updating the Q-value of action $$a_t$$ from state $$S_t$$. Taking this action will result in moving to $$S_{t+1}$$. Taking action $$a_t$$ from state $$S_t$$ results in a reward $$r_t$$. $$\alpha$$ is known as the learning rate, and controls how much Q-values can change based on a single observation (usually $$\alpha$$ is pretty small). $$\lambda$$ is a parameter that controls how much future state Q-values influence the current one. Therefore the Q-value of $$a_t$$ (from state $$S_t$$) is updated by taking a weighted sum of its current value, and the combination of the reward obtained plus the best Q-value possible from the state reached by performing $$a_t$$.

## Tic-tac-toe

To experiment with this, I've chosen to implement Q-learning to learn a strategy for playing tic-tac-toe (noughts and crosses). To keep things simple, I'll assume that the agent being trained is always player 1 (always goes first).

To implement this in python, we can use [`kaggle-environments`](https://github.com/Kaggle/kaggle-environments), a pythonm package that includes a handy tic-tac-toe environment.

The tic-tac-toe environment stores the table as a list of integers. Each entry in the list can take one of three values, 0, 1, or 2.