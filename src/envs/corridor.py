from enum import Enum

from numpy.random import choice, binomial
from numpy import arange

import gym
from gym import spaces
from gym.utils import seeding

# Action "Enum"
LEFT = 0
RIGHT = 1

class Corridor(gym.Env):
    """Corridor environment
    This game is allows the reproduction of the example used to explain 
    policy gradients in Sutton on page 323 and the chain example used by 
    Osband.
    * N: The environment consists of a N long corridor where the agent can move
     left or right.
    * K: In a K number of states, the direction traveled when given is the
     opposite of the expected. I.e. action left will cause the agent to move right.
    * p: Probability of success when moving right. 1-p probability of moving left
      instead.
    * Reaching the goal gives a reward of +1
        
    Code built from the Chain environment in AI GYM
    """

    def __init__(self, N=3, K=0, p=1):
        self.seed()
        self.N = N
        
        self.reverse_states = choice(arange(N), size=K, replace=False)
        self.p = p

        self.state = 0  # Start at beginning of the chain
        self.steps = 0
        self.max_steps = N

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.N)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self, action):
        assert self.action_space.contains(action)

        action = self.env_changes_to_actions(action)
        self.transition(action)
        reward = self.reward_calculator()

        if self.steps > self.max_steps:
            done = True
        else:
            done = False

        self.steps += 1

        return self.state, reward, done, {}

    def env_changes_to_actions(self, action):

        # If in a reverse state swap action.
        if self.state in self.reverse_states: 
            action = 1 - action

        # If trying to move right there is a prob of moving left
        if action == RIGHT:
            action = binomial(1, p=self.p) # p prob of right

        return action

    def transition(self, action):

        if action == LEFT:
            if self.state != 0: 
                self.state -= 1

        elif action == RIGHT and self.state < self.N - 1:  # 'forwards action'
            self.state += 1

    def reward_calculator(self):

        if self.state == self.N - 1:
            reward = 1
        else:
            reward = 0
            
        return reward

    def reset(self):
        self.state = 0
        self.steps = 0

        return self.state