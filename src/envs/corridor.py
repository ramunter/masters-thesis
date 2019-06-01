from enum import Enum

from numpy.random import choice, binomial
from numpy import arange, zeros, array

import gym
from gym import spaces
from gym.utils import seeding

# Action "Enum"
LEFT = 0
RIGHT = 1


class Corridor(gym.Env):
    """
    Corridor environment.  
    This game is allows the reproduction of the example used to explain 
    policy gradients in Sutton on page 323 and the chain example used by 
    Osband. Reaching the goal gives a reward of +1.  

    args:  
        N: The environment consists of a N long corridor where the agent can
           move left or right.  
        K: In a K number of states, the direction traveled when given is the
           opposite of the expected. I.e. action left will cause the agent to
           move right.    
        p: Probability of success when moving right. 1-p probability of moving
           left instead.  

    Code built based on the Chain environment in AI GYM
    """

    def __init__(self, N=3, K=0, p=1):
        self.seed()
        self.N = N

        self.reverse_states = choice(arange(N), size=K, replace=False)
        self.p = p

        self.state = 1  # Start at beginning of the chain
        self.steps = 1
        self.max_steps = N

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.N)

    @property
    def state_output(self):
        return array([self.steps==self.state, self.state])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        self.steps += 1

        action = self.env_changes_to_actions(action)
        self.transition(action)
        reward = self.reward_calculator(action)

        if self.steps >= self.max_steps:
            done = True
        else:
            done = False


        return self.state_output, reward, done, {}

    def env_changes_to_actions(self, action):

        # If in a reverse state swap action.
        if self.state in self.reverse_states:
            action = 1 - action

        # If trying to move right there is a prob of moving left
        if action == RIGHT:
            action = binomial(1, p=self.p)  # p prob of right

        return action

    def transition(self, action):

        if action == LEFT:
            if self.state != 1:
                self.state -= 1

        elif action == RIGHT and self.state < self.N:  # 'forwards action'
            self.state += 1

    def reward_calculator(self, action):

        if self.state == self.N:
            reward = 1
        elif action == 0:
            reward = 1/(100*self.N)
        else:
            reward = 0

        return reward

    def reset(self):
        self.state = 1
        self.steps = 1

        return self.state_output

    