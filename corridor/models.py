from numpy.random import binomial

class Actor():

    def __init__(self, initial_param=0.5, lr=0.1):

        self.param = initial_param
        self.lr = lr

    def policy(self, state):
        action = binomial(1, self.param)
        return action

    def update(self, q_value, action):
        self.param += self.lr*q_value*(1/self.param) # 1/w is the derivative of ln(policy(action))
    
class Critic():

    def __init__(self, lr=0.1):
        self.param = -5
        self.param2 = -5
        self.const = 1
        self.lr = lr

    def q_value(self, state, action):
        return action*self.param + state*self.param2 + self.const

    def best_action(self, state):
        left_value = self.q_value(state, 0)
        right_value = self.q_value(state, 1)
        if left_value > right_value:
            return 0
        return 1

    def update(self, r, q_value, next_q_value):
        # q = w*s + w2*a + c
        # q' = w w.r.t s 
        self.param += self.lr*(r + next_q_value - q_value)*self.param
        self.param2 += self.lr*(r + next_q_value - q_value)*self.param2
        # q' = 1 w.r.t c
        self.const += self.lr*(r + next_q_value - q_value)*1

class Optimal():
    # Class to test that things are working as expected
    def policy(self):
        return 1
