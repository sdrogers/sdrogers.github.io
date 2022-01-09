---
layout: post
author: Simon
title: Reinforcement Learning
---

# States, Agents, and Actions

Many useful tasks can be expressed using the language of _states_ and _agents_. An _agent_ exists in some world, the configuration of which at some time instant is known as its _state_. _Agents_ can perform _actions_ which might change the _state_, or move them to another _state_. In such tasks our goal will often be to come up with a strategy for the _agent_ that determines the optimal _action_ to take in any particular _state_ to help solve the task at hand.

To make this less abstract, consider the game of chess. The _agent_ is playing against some other player (could be a human, could be another ar tificial agent). At any point in the game, the _state_ is the current configuration of pieces on the board. The possible _actions_ are the moves that the _agent_ can make and the task is to try and win. Another example is a driverless car: at any time instant, the state is determined by the data coming from all of its sensors (cameras, microphones, etc), _actions_ would be steering, accelerating, brakeing, etc and the task might be to get from one location to another (safely).

## Reinforcement Learning

Reinforcement Learning (RL) is a family of techniques developed to train agents to perform well in tasks like this. Agents that initially behave randomly are allowed to attempt the task many times. Reward values are provided depending on how well they do, and these reward values are used to positively and negatively reinforce the actions that led to them. For example, in a game of chess, a high positive reward might be provided if the agent wins the match and the agent is updated to that the choices that led to the win are more likely in the future. If the agent loses, a large negative reward might be used to update the agent to make the moves that led to defeat less likely.