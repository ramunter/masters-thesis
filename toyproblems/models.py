from numpy.random import binomial

class Actor():

    def __init__(self, initial_param=0.5):

        self.param = initial_param

    def policy(self, state):
        action = binomial(1, self.param)
        return action

    def update(self, q_value, action):
        self.param += lr*q_value*(s)


        ln w*x  = w/wx

class Critic():

    def __init__(self, )
