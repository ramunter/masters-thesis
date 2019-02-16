                    
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
                    array([self.x, self.y])
                    (2, 1))

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

    def __init__(self, N=3):
        self.seed()
        self.N = N

        self.is_reverse_state = binomial(1, p=0.5, size=(N,N))

        self.state = State(0,0)  # Start at beginning of the chain
        self.steps = 0
        self.max_steps = N

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiDiscrete((self.N, self.N))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def action_state_handler(self, action):
        if self.is_reverse_state[self.state.x, self.state.y]:
            action = 1 - action
        return action

    def step(self, action):
        assert self.action_space.contains(action)

        action = self.action_state_handler(action)
        self.transition(action)
        reward = self.reward_calculator()

        if self.steps > self.max_steps:
            done = True
        else:
            done = False

        self.steps += 1

        return self.state, reward, done, {}

    def transition(self, action):

        if action == LEFT:
            if self.state.x != 0:
                self.state.x -= 1

        elif action == RIGHT and self.state.x < self.N - 1:  # 'forwards action'
            self.state.x += 1

        self.state.y += 1

    def reward_calculator(self):

        if self.state.x == self.N - 1:
            reward = 1

        else:
            reward = -(1/self.N * 0.1)

        return reward

    def reset(self):
        self.state = State(0,0)
        self.steps = 0

        return self.state
