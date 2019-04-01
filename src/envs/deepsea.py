
from numpy.random import binomial
from numpy import array, reshape

import gym
from gym import spaces
from gym.utils import seeding

# Action "Enum"
LEFT = 0
RIGHT = 1


class State():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def as_array(self):
        return reshape(
            array([self.x, self.y, self.x*self.y, self.x**2, self.y**2]),
            (5, 1))


class DeepSea(gym.Env):
    """
    DeepSea environment.

    This game is based on the deep sea environment introduce by Osband. Reaching
    the goal gives a reward of +1.  
        N: The environment consists of a NxN grid where the agent can move left or right.  

    Code built from the Chain environment in AI GYM.
    """

    def __init__(self, N=3):
        assert N > 1, "DeepSea has a minimum size of 2"

        self.seed()
        self.N = N

        self.is_reverse_state = binomial(1, p=0.5, size=(N, N))

        self.state = State(0, 0)  # Start at beginning of the chain
        self.steps = 1
        self.max_steps = N

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiDiscrete((self.N, self.N))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        self.steps += 1

        self.transition(action)
        reward = self.reward_calculator()

        if self.steps >= self.max_steps:
            done = True
        else:
            done = False

        return self.state.as_array, reward, done, {}

    def transition(self, action):

        action = self.action_state_handler(action)

        if action == LEFT:
            if self.state.x != 0:
                self.state.x -= 1

        elif action == RIGHT and self.state.x < self.N - 1:  # 'forwards action'
            self.state.x += 1

        self.state.y += 1

    def action_state_handler(self, action):
        if self.is_reverse_state[self.state.x, self.state.y]:
            action = 1 - action
        return action

    def reward_calculator(self):

        if self.state.x == self.N - 1:
            reward = 1
        else:
            reward = 0
        return reward

    def reset(self):
        self.state = State(0, 0)
        self.steps = 1

        return self.state.as_array
