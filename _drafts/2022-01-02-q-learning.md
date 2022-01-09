---
layout: post
author: Simon
title: Noughts and Crosses with Q-learning
---

Reinforcement Learning is something I've been keen to get to understand a bit more for a while. Ultimately, my goal is to have a play with deep Q-learning. As a first step, I decided to have a play with plain Q-learning.



## Q-learning

Q-learning is a popular method within the field of Reinforcement Learning (RL). The goal of RL is to train an _agent_ to be able to navigate through a state space in an optimal way. As an example, I'm using the game of tic-tac-toe (aka noughts and crosses).

The _agent_ is one of the tic-tac-toe players. In everything below, it plays against another computer player although there's no reason it couldn't play against a human. The goal of RL is to train the agent such that it can make sensible moves based on the current board situation (the state).

Whereas the standard ML paradigm is to train models from a static training set (i.e. collect training data -> train -> predict), in RL the training is done via the _agent_ actually performing the task, over and over (and over and over and over) again.

Consider our agent at some stage in the game. The _state_ is determined by the current orientation of noughts and crosses on the board. The _agent_ can choose one of a fixed set of actions (where to place their marker). Q-learning attempts to learn a mapping between state, action pairs and a Q-value that is indicative of how good the action is in that state.

To make this more concrete, let's use the numbers 1 to 9 to define the positions on the board:

1 2 3
4 5 6
7 8 9

And, assume that the current state of the board is (where 1 and 2 stand for players 1 and 2 respectively):

1 0 2
0 1 2
0 0 0

If our agent is player 1, then there are 5 possible actions (positions 2, 4, 7, 8, and 9). A trained Q-learner would include a Q-table that held a score for each of these actions _from_ this particular state. In this example, we'd hope that for this state, it would have the highest score for action 9, which is obviously the best move to make.

The Q-learner keeps track of all these Q-values within a Q-table. As they have to store scores for all valid state, action pairs, they can become enormous. In Python, I will use a dictionary to store the Q-table, where the key will be a String representation of the current board (state). For example, the board state shown above could be stored as a list:

```python
state = [1, 0, 2, 0, 1, 2, 0, 0, 0]
state_string = "102012000"
```

The Q-table would have one key, value pair for each possible state. In my implementation I encode the value in the dictionary as a list of Q-values for this state. E.g. for this state:

```python
q_table["102012000"] = [0, 1, 0, 0.5, 0, 0, 2.0, 0.5, 23.2]
```
Suggesting that action 9 (bottom right corner) is the best action. Note that in my implementation I always define the action list as having length 9 despite the fact that some moves are invalid. This doesn't really matter as long as we implement the agent to choose the _valid_ move with the highest Q-value and not just the move with the highest Q-value.

So, where's the learning? As mentioned above, RL methods learn by exploring the space over and over again. For tic-tac-toe, this means playing a _lot_ of games. In order for the method to _learn_, it needs to know when it has made a good move. In RL this is done through _rewards_. In particular, for tic-tac-toe we will give a high positive reward when the agent wins, and a high negative reward when it loses (0 for a draw)


## Updating Q-values

At the heart of Q-learning is how we update the values in the Q-table. This is done according to Bellman's equation:

$$ Q_{new}(S_t, a_t) = (1-\alpha) Q(S_t, a_t) + \alpha(r_t + \lambda \max_a Q(S_{t+1}, a)) $$

Let's look at the various parts of this. Bellman's equation is for updating the Q-value of action $$a_t$$ from state $$S_t$$. Taking this action will result in moving to $$S_{t+1}$$. Taking action $$a_t$$ from state $$S_t$$ will also result in the agent receiving a reward $$r_t$$. 


TODO

$$\alpha$$ is known as the learning rate, and controls how much Q-values can change based on a single observation (usually $$\alpha$$ is pretty small). $$\lambda$$ is a parameter that controls how much future state Q-values influence the current one. Therefore the Q-value of $$a_t$$ (from state $$S_t$$) is updated by taking a weighted sum of its current value, and the combination of the reward obtained plus the best Q-value possible from the state reached by performing $$a_t$$.


### Limitations

A couple of limitations emerge from this description of Q-learning. Firstly, for many problems, the state space is enormous and the Q-table is going to become really big. Secondly, assuming that one needs to visit each stage multiple times during training to obtain reliable Q-values suggests that training is going to take a _long_ time. Deep Q-learning attempts to overcome this: rather than keep a table of all Q-values, it trains a neural network to _predict_ Q-values for states -- more on this in another post once I've had a go with it.



## Tic-tac-toe

To experiment with this, I've chosen to implement Q-learning to learn a strategy for playing tic-tac-toe (noughts and crosses). To keep things simple, I'll assume that the agent being trained is always player 1 and always goes first.

To implement this in python, we can use [`kaggle-environments`](https://github.com/Kaggle/kaggle-environments), a pythonm package that includes a handy tic-tac-toe environment.

The tic-tac-toe environment stores the table as a list of integers. Each entry in the list can take one of three values: 0, 1, or 2. The first three elements in the list correspond to the first row, etc. For example, the list:
```python
[1, 2, 0, 2, 0, 0, 0, 1, 0]
```
corresponds to the following board:
```
1 2 0
2 0 0
0 1 0
```

`kaggle-environments` provides handy methods to help train and run simulations. For example, the following code will run a game between two of the players supplied by kaggle -- random, and reaction -- and then render the resulting game.

```python
from kaggle-environments import make
env = make('tic-tac-toe')
env.run(['randon', 'reaction'])
env.render(mode='ipython')
```

### Building an agent

Our first step is to build an agent that is compatible with `kaggle-environments`. This takes the form of a method that returns an action given the current state observation. For `tic-tac-toe`, the board is accessible via `observation.board`. Our agent code is straightforward and looks like this:

```python
def ttt_agent(observation, configuration):
    action = get_action(observation.board)
    return action
```

The work is done via the following `get_action` method, that is passed the current state of the game (the board):

```python
def get_action(board):
    '''
    Method to get an action based on the current board
    Uses epsilon-greedy exploration: an action is chosen uniformly randomly
    with probability EPSILON, and chosen as the action with highest Q-value
    with probability 1-EPSILON
    '''
    method = np.random.choice(['random', 'best'], p=[EPSILON, 1-EPSILON])
    str_board = "".join([str(a) for a in board])
    if method == 'random' or (str_board not in q_table):
        # make a random choice if random is chosen or the state
        # doesn't (yet) exist in the q-table
        action = random_move(board)
    else:
        # choose the max permissible from the q-table
        action = best_move(board, q_table[str_board])
    return int(action)
```
This method decides (based on $$\epsilon$$) whether to choose a random action, or choose the action that has highest Q-value for the current state. Note that if the state doesn't exist in the Q-table (meaning it is yet to be visited) a random move is chosen.

The two methods called here are defined as follows:

```python
import numpy as np
def random_move(board):
    '''
    Picks a _valid_ action randomly
    '''
    N = len(board)
    probs = np.ones(N)
    # set invalid action probabilities to zero
    probs[np.array(board) > 0] = 0
    # renormalise
    probs /= probs.sum()
    # sample and return
    return np.random.choice(range(N), p=probs)
```
and
```python
def best_move(board, q_vals):
    '''
    Picks the _valid_ move with the highest Q-value.
    Ties are broken randomly/
    '''
    mq = min(q_vals)
    q_vals = np.array(q_vals)
    q_vals[np.array(board) > 0] = mq - 1
    action = np.random.choice(np.flatnonzero(q_vals == q_vals.max()))
    return action
```
The `get_action` method assumes that a `q_table` exists. An initial test that things work ok is to make sure a game can be played between `ttt_agent` and one of the pre-defined agents. E.g.

```python
q_table = {} # empty q_table will result in random play
env.run([ttt_agent, 'random'])
env.render(mode='ipython')
```

### Training

Our training process will involve updating the Q-table after each complete game. We therefore need to be able to play a game, storing each state and all actions. The following class will be useful for storing the states etc within a single game:

```python
class Sequence(object):
    def __init__(self):
        self.clear_sequence()
    
    def clear_sequence(self):
        # reset everything
        self.states = [] # list of states
        self.rewards = [] # list of rewards
        self.actions = [] # list of actions
    
    def add_state_to_sequence(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)
    
    def rotate(self):
        # return a sequence rotated by 1 turn to the right
        rotated_sequence = Sequence()
        for i, state in enumerate(self.states):
            rotated_state = Sequence.rotate_state(state)
            rotated_action = Sequence.rotate_action(self.actions[i])
            rotated_sequence.add_state_to_sequence(
                rotated_state,
                rotated_action,
                self.rewards[i]
            )
        return rotated_sequence

    @staticmethod
    def rotate_state(state):
        '''
        Single rotation to the right for the state
        '''
        new_idx = [6, 3, 0, 7, 4, 1, 8, 5, 2]
        return np.array(state)[new_idx]
    
    @staticmethod
    def rotate_action(action):
        '''
        Single rotation to the right for an action
        '''
        new_idx = [2, 5, 8, 1, 4, 7, 0, 3, 6]
        return new_idx[action]
```
As well as `run`, the `kaggle-environments` environment provides `train`. This class allows code to manually step through a game. Creating a training session against the `reaction` agent is done as:
```python
trainer = train([None, 'reaction'])
```
Using such a trainer, the following method will run a single game:
```python
def single_game(env, trainer, player=0):
    '''
    Play a single game
    (uses global EPSILON and q_table vars)
    '''
    sequence = Sequence()
    current_board = trainer.reset()['board']
    while True:
        action = get_action(current_board)
        old_board = current_board.copy()
        # take a single step in the game. We provide our move (action) and then
        # the other agent takes theirs
        next_observation, dummy, overflow, info = trainer.step(action)
        current_board = next_observation['board']

        # check the status to see if the game has finished or not
        if env.state[0].status == 'DONE':
            # env.state[player].reward will be 1 if we won, and -1 otherwise
            reward = 20 * env.state[player].reward
            finished = True
        else:
            reward = 0
            finished = False
        
        # store the state, action and reward
        sequence.add_state_to_sequence(old_board, action, reward)
        
        if finished:
            break
            
    return sequence
```
Here we pass the `kaggle-environment` (`env`), the `trainer` and, optionally the player (which will allow us to play as player 0 or player 1).

The final thing we need is to update the Q-table based on the `sequence` returned from this method. To do this, we work backwards through the sequence, applying Bellman's equation to update the Q-value for all of the state, action pairs visited in the game:

```python
def update_q_table(q_table, sequence, alpha=0.1, lamb=0.7):
    # make sure all states exist in table, adding them if not
    for s in sequence.states:
        str_s = "".join([str(a) for a in s])
        if str_s not in q_table:
            q_table[str_s] = [0,0,0,0,0,0,0,0,0]
        
    # loop over states in reverse:
    for i in range(len(sequence.states) - 1, -1, -1):
        str_s = "".join([str(a) for a in sequence.states[i]])
        action = sequence.actions[i]
        reward = sequence.rewards[i]

        if i == len(sequence.states) - 1:
            # final state is treated differently as it has no possible
            # _next_ state
            next_max_q = 0
        else:
            # otherwise find the state we got to by perfoming the action
            next_str_s = "".join([str(a) for a in sequence.states[i + 1]])
            next_max_q = max(q_table[next_str_s])
        
        # update Q-value
        q_table[str_s][action] = (1 - alpha) * q_table[str_s][action] \
            + alpha * (reward + lamb * next_max_q)
        
    return q_table
```

This method contains the actual _learning_. We start by checking that all of the states we visited are in the Q-table. If not, we add them with default Q-values (all zeros).

We then loop over the states in reverse. The first one we look at (the final state in the game) has to be treated slightly differently: it has no _next_ state, so the final term in Bellman's equation will be zero. For other states, it is computed as the maximum Q-value in the state reached when the relevant action is applied. Once we have this value, we can use Bellman's equation to update the single Q-value corresponding to the current state, action pair.

Finally, we can put this whole process into a loop to play lots of games.

```python
q_table = {} # Empty Q-table
ALPHA = 0.1 # Learning rate
LAMBDA = 0.7 # Reward discount
N_GAME = 5000 # Total number of games
EPSILON = 1  # Initial value of epsilon
N_EXPLORE = 2000 # Number of games before starting EPSILON decay
PLAYER = 0 # Which player we are
EE = 1e-5 # Target final EPSILON value

# Compute EPSILON discount factor
eps_fac = np.exp(np.log(EE) / (N_GAME - N_EXPLORE))

# Create train object
if PLAYER == 0:
    trainer = env.train([None, 'reaction'])
else: 
    trainer = env.train(['reaction', None])

for game in range(N_GAME):
    if game % 50 == 0:
        print(f'Game {game}, eps {EPSILON}, n_s {len(q_table)}')
    
    # If we're past the initial exploration phase
    if game > N_EXPLORE:
        EPSILON *= eps_fac
    
    # Run a single game
    sequence = single_game(env, trainer, player=PLAYER)

    # Update the q_table, including doing it with the sequence rotated
    q_table = update_q_table(q_table, sequence, alpha=ALPHA, lamb=LAMBDA)
    for i in range(3):
        sequence = sequence.rotate()
        q_table = update_q_table(q_table, sequence, alpha=ALPHA, lamb=LAMBDA)
```
Two things should be noted here. Firstly, I have implemented a slight change to the epsilon-greedy exploration strategy in that $$\epsilon$$ doesn't decay at all for `N_EXPLORE` iterations. After this, it decays exponentially (by multiplying by a constant factor). The factor is chosen so that at the final game, `EPSILON = EE = 1e-5`. Secondly, we make use of the fact that tic-tac-toe is invariant to rotations of the board. I.e. a particular game is essentially equivalent to another game where the board has been rotated by 90, 180, or 270 degrees. We make use of this by updating the Q-table three times for each game -- with the original sequence, and then sequences rotated once, twice, and three times.