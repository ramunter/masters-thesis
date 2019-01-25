from numpy.random import binomial
from numpy import array
from sklearn.linear_model import SGDRegressor

import tensorflow as tf
import edward as ed
from edward.models import Normal

def featurizer(state, action):
    return array([state, action]).reshape(1,-1)
    #return array([state, action, state*action, state**2, action**2]).reshape(1, -1) # Reshape to work with sklearn...


class MoveRight():
    # Class to test that things are working as expected
    def policy(self):
        return 1


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

    def get_action(self, state):

        if binomial(1,0.2):
            return binomial(1, 0.5)
        else:
            # Perform step
            return self.best_action(state)

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


class PosteriorCritic():

    def __init__(self, features_shape, gamma=0.9):
        create_model()

    def create_model(self):
        self.features = tf.placeholder(tf.int32, [1, features_shape])
        self.w = Normal(loc=tf.zeros(state_shape), scale=tf.ones(features_shape))
        self.b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
        self.y = Normal(loc=ed.dot(features, w) + b, scale=tf.ones(1))
        self.post_w = Normal(loc=tf.get_variable("qw/loc", [features_shape]),
                    scale=tf.nn.softplus(tf.get_variable("qw/scale", [features_shape])))
        self.post_b = Normal(loc=tf.get_variable("qb/loc", [1]),
                    scale=tf.nn.softplus(tf.get_variable("qb/scale", [1])))


    def update(self, state, action, reward, next_q_value, done):
        target = reward
        if not done:
            target += self.gamma*next_q_value

        features = featurizer(state, action)
        target = array([target]).reshape((1,))



class Actor():

    def __init__(self, initial_action_param=0.5, lr=0.1):

        self.action_param = initial_action_param
        self.lr = lr

    def policy(self):
        return 1

