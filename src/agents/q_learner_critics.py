from collections import namedtuple

import numpy as np
from numpy.random import binomial

from sklearn.linear_model import SGDRegressor

from src.agents.q_learner import CriticTemplate
from src.agents.util import featurizer, GaussianRegression

class EGreedyCritic(CriticTemplate):
    """ A regular E greedy agent with a constant E.
    """
    
    def __init__(self, eps=0.2, lr=0.01):
        """ Initializes a linear model.

        args:
            eps (float): Probability of choosing a random action.
            lr (float): Learning rate used by the linear model. 
        
        """
        self.eps = eps

        self.model = SGDRegressor(learning_rate="constant", eta0=lr)
        features = featurizer(state, action=0)
        self.model.partial_fit(features, np.array([0])) # SKlearn needs partial fit to be run once before use

    def get_action(self, state):
        """ Gets an action using the E greedy approach.

        args:
            state (np.array)
        
        returns:
            action (int)
        """

        if binomial(1,0.2):
            return binomial(1, 0.5)
        else:
            return self.best_action(state)

    def get_target_action_and_q_value(self, state):
        """ Calculates the optimal action and Q-value

        args:
            state (np.array)

        returns:
            action (int)
            q-value (float)
        """
        left_value = self.q_value(state, 0)
        right_value = self.q_value(state, 1)
        if left_value > right_value:
            return 0, left_value
        return 1, right_value

    def update(self, state, action, target):
        """ Takes one optimzation step for the linear model.

        args;
            state (np.array)
            action (int)
            target (float): Target Q-value.
        """
        features = featurizer(state, action)
        self.model.partial_fit(features, target)

    def q_value(self, state, action):
        """ Caclulates Q-value given state and actio

        args:
            state (np.array)
            action (int)

        returns:
            q-value (float)

        """
        features = featurizer(state, action)
        return self.model.predict(features)[0]

    def print_parameters(self):
        print('Coefficients: \n', self.model.coef_)


class UBECritic(CriticTemplate):
    """ The uncertainty bellman equation method without propagating uncertainty
    """

    def __init__(self, lr=0.01):

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

    def action_variance(self, state, action):
        var_action = state*self.sigma[action]*state

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
        var_q = self.action_variance(state, action)
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
    """ Same method as TargetUBECritic but samples the target Q aswell.
    """

    def get_target_action_and_q_value(self, state):
        Q_left = self.sample_q(state, 0)
        Q_right = self.sample_q(state, 1)
        if Q_left > Q_right:
            return 0, Q_left
        return 1, Q_right


class GaussianBayesCritic(CriticTemplate):
    """ Bayesian linear model using a gaussian prior with known variance.
    
    Samples both the Q- and target Q-value by sampling the parameters per step.
    """

    def __init__(self, lr=0.01):
                
        self.model = GaussianRegression()

    def init_model(self, state):
        pass

    def get_action(self, state):
        action, _ = self.get_target_action_and_q_value(state)
        return action

    def get_target_action_and_q_value(self, state):
        coef = self.sample_coef()
        Q_left = self.q_value(state, 0, coef)
        Q_right = self.q_value(state, 1, coef)
        if Q_left > Q_right:
            return 0, Q_left
        return 1, Q_right

    def update(self, state, action, target):

        if type(state) == list:
            X = np.array([state, action, [1]*len(state)]).T
            target =  np.repeat(np.array(target, ndmin=2), repeats=len(state), axis=0)

        else:
            X = featurizer(state, action)
            X = np.append(X, [[1]], axis=1) # add constant

        inv_cov = np.linalg.inv(self.model.cov)
        self.model.mean = np.linalg.inv(X.T @ X + self.model.noise * inv_cov) @ \
            (X.T @ target + inv_cov@self.model.mean)
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
    """ Bayesian linear model using a gaussian prior with known variance.
    
    Samples both the Q- and target Q-value by sampling the parameters per episode.
    """

    def __init__(self, lr=0.01):
                
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


