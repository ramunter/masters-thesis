from numpy.random import choice
from numpy import arange

import gym
from gym import spaces
from gym.utils import seeding

class Corridor(gym.Env):
    """Corridor environment
    This game is a reproduction of the example used to explain policy gradients in Sutton on page 323.
    * The environment consists of a N long corridor where the agent can move left or right. 
    * In a K number of states, the direction traveled when given is the opposite of the expected. I.e. action left will cause the agent to move right.
    * The states are identical meaning the policy has to a probabilistic choice between acting left or right.
    * Reaching the goal gives a reward of +1
    
    Code built from the Chain environment in AI GYM
    """

    def __init__(self, N=3, K=1):
        self.seed()
        self.n = N
        self.reverse_states = choice(arange(N), size=K, replace=False)
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def action_state_handler(self, action):
        if self.state in self.reverse_states:
            action = 1 - action
        
    def step(self, action):
        assert self.action_space.contains(action)
        reward=0
        done = False

        if self.state in self.reverse_states: # If in a reverse state swap action.
            action = 1 - action

        if action == 0 and self.state != 0:  # 'backwards action'
            self.state -= 1

        elif self.state < self.n - 1:  # 'forwards action'
            self.state += 1
        else:
            assert False, "In final or past final state. This should never occur."

        if self.state == self.n - 1:
            reward = 1
            done= True

        return 0, reward, done, {}

    def reset(self):
        self.state = 0
        return self.state
