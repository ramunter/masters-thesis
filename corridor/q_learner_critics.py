from collections import namedtuple

import numpy as np
from numpy.random import binomial

from sklearn.linear_model import SGDRegressor

from q_learner import CriticTemplate
from util import featurizer, GaussianRegression

class EGreedyCritic(CriticTemplate):

    def __init__(self, lr=0.01, gamma=0.9):

        self.model = SGDRegressor(learning_rate="constant", eta0=lr)
    
    def init_model(self, state):
        features = featurizer(state, 0)
        self.model.partial_fit(features, np.array([0])) # SKlearn needs partial fit to be run once before use

    def get_action(self, state):

        if binomial(1,0.2):
            return binomial(1, 0.5)
        else:
            return self.best_action(state)

    def get_target_action_and_q_value(self, state):
        left_value = self.q_value(state, 0)
        right_value = self.q_value(state, 1)
        if left_value > right_value:
            return 0, left_value
        return 1, right_value

    def update(self, state, action, target):       
        features = featurizer(state, action)
        self.model.partial_fit(features, target)

    def q_value(self, state, action):
        features = featurizer(state, action)
        return self.model.predict(features)[0]

    def print_parameters(self):
        print('Coefficients: \n', self.model.coef_)

class UBECritic(CriticTemplate):

    def __init__(self, lr=0.01, gamma=0.9):

        self.model = SGDRegressor(learning_rate="constant", eta0=lr)
        self.sigma = [np.eye(1)]*2 # 2 is num actions
        self.beta = 6
    
    def init_model(self, state):
        features = featurizer(state, 0)
        self.model.partial_fit(features, np.array([0])) # SKlearn needs partial fit to be run once before use

    def get_action(self, state):
        return self.sample_action(state)

    def get_target_action_and_q_value(self, state):
        left_value = self.mean_q_value(state, 0)
        right_value = self.mean_q_value(state, 1)
        if left_value > right_value:
            return 0, left_value
        return 1, right_value

    def mean_q_value(self, state, action):
        features = featurizer(state, action)
        return self.model.predict(features)[0]

    def action_variance(self, features):
        action = features[0,-1]
        features = features[0,:-1]
        var_action = features*self.sigma[action]*features

        return(var_action)

    def sample_action(self, state):
        Q_left = self.sample_q(state, 0)
        Q_right = self.sample_q(state, 1)
        if Q_left > Q_right:
            return 0    
        return 1

    def sample_q(self, state, action):
        features = featurizer(state, action)
        mean_q = self.model.predict(features)[0]
        var_q = self.action_variance(features)
        sample = np.random.standard_normal(size=1)
        
        sample_q = mean_q + self.beta*sample*(var_q**0.5)
        return sample_q

    def update(self, state, action, target):
        features = featurizer(state, action)
        self.model.partial_fit(features, target)
        self.update_sigma(features)

    def update_sigma(self, features):
        
        action = features[0,-1]
        features = features[0,:-1]

        sigma = self.sigma[action]
        change_numerator = sigma * features * \
                                features * sigma

        change_denominator = 1 + features * sigma * features
        self.sigma[action] -= change_numerator/change_denominator

    def print_parameters(self):
        print('Coefficients: \n', self.model.coef_)


class SampleTargetUBECritic(UBECritic):

    def get_target_action_and_q_value(self, state):
        Q_left = self.sample_q(state, 0)
        Q_right = self.sample_q(state, 1)
        if Q_left > Q_right:
            return 0, Q_left
        return 1, Q_right

class GaussianBayesCritic(CriticTemplate):

    def __init__(self, lr=0.01, gamma=0.9):
                
        self.model = GaussianRegression()

    def init_model_params(self, dim=3):
        GaussianParameters = namedtuple('Parameters', ['mean', 'cov', 'noise'])
        params = GaussianParameters(mean=np.zeros((dim,1)), cov=np.eye(dim), noise=1)
        return params

    def init_model(self, state):
        pass

    def get_action(self, state):
        action, q_value = self.get_target_action_and_q_value(state)
        return action

    def get_target_action_and_q_value(self, state):
        coef = self.sample_coef()
        Q_left = self.q_value(state, 0, coef)
        Q_right = self.q_value(state, 1, coef)
        if Q_left > Q_right:
            return 0, Q_left
        return 1, Q_right

    def update(self, state, action, target):
        X = featurizer(state, action)
        inv_cov = np.linalg.inv(self.model.cov)
        X = np.append(X, [[1]], axis=1) # add constant
        self.model.mean = np.linalg.inv(X.T @ X + self.model.noise * inv_cov) @ \
            (X.T * target + inv_cov@self.model.mean)
        self.model.cov = np.linalg.inv(self.model.noise**(-2) * X.T @ X + inv_cov)
        
    def sample_coef(self):
        coef = np.random.multivariate_normal(self.model.mean[:,0], self.model.cov)
        return coef 
    
    def q_value(self, state, action, coef):
        features = featurizer(state, action)
        features = np.append(features, [[1]], axis=1) # add constant
        return features@coef

    def print_parameters(self):
        print("Coefficients")
        print("Mean:\n", self.model.mean)
        print("Cov:\n", self.model.cov)

class DeepGaussianBayesCritic(GaussianBayesCritic):
    """
    Deep exploration
    """

    def __init__(self, lr=0.01, gamma=0.9):
                
        self.model = GaussianRegression()
        self.coef = self.sample_coef()

    def get_target_action_and_q_value(self, state):
        Q_left = self.q_value(state, 0)
        Q_right = self.q_value(state, 1)
        if Q_left > Q_right:
            return 0, Q_left
        return 1, Q_right

    def reset(self):
        self.coef = self.sample_coef()

    def sample_coef(self):
        coef = np.random.multivariate_normal(self.model.mean[:,0], self.model.cov)
        return coef
    
    def q_value(self, state, action):
        features = featurizer(state, action)
        features = np.append(features, [[1]], axis=1) # add constant
        return features@self.coef

    def print_parameters(self):
        print("Coefficients")
        print("Mean:\n", self.model.mean)
        print("Cov:\n", self.model.cov)


