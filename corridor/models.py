from numpy.random import binomial

class Actor():

    def __init__(self, initial_action_param=0.5, lr=0.1):

        self.action_param = initial_action_param
        self.lr = lr

    def policy(self, state):
        action = binomial(1, self.action_param)
        return action

    def update(self, q_value, action):
        self.action_param += self.lr*q_value*(1/self.action_param) # 1/w is the derivative of ln(policy(action))
    
class Critic():

    def __init__(self, lr=0.1, gamma=0.9):
        self.action_param = 5
        self.state_param = 5
        self.const = 1
        self.lr = lr
        self.gamma = gamma
        

    def q_value(self, state, action):
        return action*self.action_param + state*self.state_param + self.const

    def best_action(self, state):
        left_value = self.q_value(state, 0)
        right_value = self.q_value(state, 1)
        if left_value > right_value:
            return 0
        return 1

    def update(self, r, q_value, next_q_value, done):
        # q = w*s + w2*a + c
        # q' = w w.r.t s 
        target = r
        if not done:
            target += self.gamma*next_q_value

        self.action_param += self.lr*(target - q_value)*self.action_param
        self.state_param += self.lr*(target - q_value)*self.state_param
        # q' = 1 w.r.t c
        self.const += self.lr*(target - q_value)*1

    def print_parameters(self):
        print("Action parameter:", self.action_param)
        print("State parameter:", self.state_param)
        print("Constant:", self.const)

    def print_q_values(self, num_states):
        print("Left ", end='')
        for state in range(0,num_states):
            print(
                  round(self.q_value(state, 0), 2),
                  end=' ')
        print()
        
        print("Right ", end='')
        for state in range(0,num_states):
            print(
                  round(self.q_value(state, 1), 2),
                  end=' ')
        print()
        

class MoveRight():
    # Class to test that things are working as expected
    def policy(self):
        return 1
