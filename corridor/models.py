from numpy.random import binomial
from numpy import array
from sklearn.linear_model import SGDRegressor

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

    def __init__(self, lr=0.01, gamma=0.9):

        self.model = SGDRegressor(learning_rate="constant", eta0=lr)
        self.gamma = gamma
    
    def init_model(self, state):
        features = featurizer(state, 0)
        self.model.partial_fit(features, array([0])) # SKlearn needs partial fit to be run once before use

    def q_value(self, state, action):
        features = featurizer(state, action)
        return self.model.predict(features)[0]

    def best_action(self, state):
        left_value = self.q_value(state, 0)
        right_value = self.q_value(state, 1)
        if left_value > right_value:
            return 0
        return 1

    def update(self, state, action, reward, next_q_value, done):
        target = reward
        if not done:
            target += self.gamma*next_q_value

        features = featurizer(state, action)
        target = array([target]).reshape((1,)) # Correct dim for SKlearn        

        self.model.partial_fit(features, target)

    def print_parameters(self):
        print('Coefficients: \n', self.model.coef_)

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
    
    def print_policy(self, num_states):
        print("Policy ", end='')
        for state in range(0, num_states):
            if self.best_action(state):
                print("r", end=' ')
            else:
                print("l", end=' ')
        print()



class MoveRight():
    # Class to test that things are working as expected
    def policy(self):
        return 1


def featurizer(state, action):
    return array([state, action]).reshape(1,-1)
    #return array([state, action, state*action, state**2, action**2]).reshape(1, -1) # Reshape to work with sklearn...