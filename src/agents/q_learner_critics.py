from collections import namedtuple

import numpy as np
from numpy.random import binomial

from sklearn.linear_model import SGDRegressor

from src.agents.q_learner import CriticTemplate
from src.agents.util import featurizer, GaussianRegression


class EGreedyCritic(CriticTemplate):
    """ A regular E greedy agent with a constant E."""

    def __init__(self, state, eps=0.2, lr=0.01):
        """
        Initializes a linear model.

        args:
            state : State from the environment of interest.
            eps   : Probability of choosing a random action.
            lr    : Learning rate used by the linear model. 
        """
        self.eps = eps
        self.model = self.setup_model(state, lr)

    def setup_model(self, state, lr):
        model = SGDRegressor(learning_rate="constant",
                             eta0=lr, fit_intercept=False)
        features = featurizer(state, action=0)
        # SKlearn needs partial fit to be run once before use
        model.partial_fit(features, np.array([0]))
        return model

    def get_action(self, state):
        """ Gets an action using the E greedy approach."""

        if binomial(1, 0.2):
            return binomial(1, 0.5)
        else:
            return self.best_action(state)

    def get_target_action_and_q_value(self, state):
        """ Calculates the optimal action and Q-value."""

        left_value = self.q_value(state, 0)
        right_value = self.q_value(state, 1)
        if left_value > right_value:
            return 0, left_value
        return 1, right_value

    def update(self, state, action, target):
        """Takes one optimization step for the linear model."""

        features = featurizer(state, action)
        self.model.partial_fit(features, target)

    def q_value(self, state, action):
        """Caclulates Q-value given state and action."""
        features = featurizer(state, action)
        return self.model.predict(features)[0]

    def print_parameters(self):
        print('Coefficients: \n', self.model.coef_)


class UBECritic(CriticTemplate):
    """ The uncertainty bellman equation method without propagating uncertainty"""

    def __init__(self, state, lr=0.01):
        """
        Initializes a linear model.

        args:
            state : State from the environment of interest.
            lr    : Learning rate used by the linear model. 
        """
        self.model = self.setup_model(state, lr)
        self.sigma = [np.eye(1)]*2  # 2 is num actions
        self.beta = 6

    def setup_model(self, state, lr):
        model = SGDRegressor(learning_rate="constant",
                             eta0=lr, fit_intercept=False)
        features = featurizer(state, 0)
        # SKlearn needs partial fit to be run once before use
        self.model.partial_fit(features, np.array([0]))
        return model

    def get_action(self, state):
        """Gets an action by sampling the Q posterior."""
        return self.sample_action(state)

    def get_target_action_and_q_value(self, state):
        """ Calculates the optimal action and Q-value based on the expected value."""
        left_value = self.mean_q_value(state, 0)
        right_value = self.mean_q_value(state, 1)
        if left_value > right_value:
            return 0, left_value
        return 1, right_value

    def mean_q_value(self, state, action):
        """Estimate the expected Q-value"""
        features = featurizer(state, action)
        return self.model.predict(features)[0]

    def action_variance(self, state, action):
        """Caclulate the Q-value variance based on the equation in the UBE paper."""
        var_action = state*self.sigma[action]*state

        return(var_action)

    def sample_action(self, state):
        """Samples action based on sampled Q-values."""
        Q_left = self.sample_q(state, 0)
        Q_right = self.sample_q(state, 1)
        if Q_left > Q_right:
            return 0
        return 1

    def sample_q(self, state, action):
        """Samples Q-value for an action."""
        features = featurizer(state, action)
        mean_q = self.model.predict(features)[0]
        var_q = self.action_variance(state, action)
        sample = np.random.standard_normal(size=1)

        sample_q = mean_q + self.beta*sample*(var_q**0.5)
        return sample_q

    def update(self, state, action, target):
        """Train the model using Q-learning TD update."""
        features = featurizer(state, action)
        self.model.partial_fit(features, target)
        self.update_sigma(features)

    def update_sigma(self, features):
        """Update the Covariance matrix."""
        action = features[0, -1]
        features = features[0, :-1]

        sigma = self.sigma[action]
        change_numerator = sigma * features * \
            features * sigma

        change_denominator = 1 + features * sigma * features
        self.sigma[action] -= change_numerator/change_denominator

    def print_parameters(self):
        print('Coefficients: \n', self.model.coef_)


class SampleTargetUBECritic(UBECritic):
    """ Same method as TargetUBECritic but samples the target Q as well."""

    def get_target_action_and_q_value(self, state):
        """Sample the target action and Q-value."""
        Q_left = self.sample_q(state, 0)
        Q_right = self.sample_q(state, 1)
        if Q_left > Q_right:
            return 0, Q_left
        return 1, Q_right


class GaussianBayesCritic(CriticTemplate):
    """
    Bayesian linear model using a gaussian prior with known variance.

    Samples both the Q- and target Q-value by sampling the parameters per step.
    """

    def __init__(self, state, lr=0.01):
        """
        Initializes a bayesian linear model.

        args:
            state : State from the environment of interest.
            lr    : Learning rate used by the linear model. 
        """
        if type(state) is int:
            feature_size = 3
        else:
            feature_size = len(state) + 2  # Add bias term and action term.

        self.model = GaussianRegression(dim=feature_size)

    def get_action(self, state):
        action, _ = self.get_target_action_and_q_value(state)
        return action

    def get_target_action_and_q_value(self, state):
        """
        Samples an action by sampling coefficients and choosing the highest
        resulting Q-value.
        """
        self.coef = self.sample_coef()
        Q_left = self.q_value(state, 0)
        Q_right = self.q_value(state, 1)
        if Q_left > Q_right:
            return 0, Q_left
        return 1, Q_right

    def update(self, state, action, target):
        """Calculate posterior and update prior."""
        if type(state) == list:
            X = np.array([state, action, [1]*len(state)]).T
            target = np.repeat(np.array(target, ndmin=2),
                               repeats=len(state), axis=0)

        else:
            X = featurizer(state, action)

        inv_cov = np.linalg.inv(self.model.cov)
        self.model.mean = np.linalg.inv(X.T @ X + self.model.noise * inv_cov) @ \
            (X.T @ target + inv_cov@self.model.mean)
        self.model.cov = np.linalg.inv(
            self.model.noise**(-2) * X.T @ X + inv_cov)

    def sample_coef(self):
        """Sample regression coefficients from the posterior."""
        coef = np.random.multivariate_normal(
            self.model.mean[:, 0], self.model.cov)
        return coef

    def q_value(self, state, action):
        """Caclulate Q-value based on sampled coefficients."""
        features = featurizer(state, action)
        return features@self.coef

    def print_parameters(self):
        print("Coefficients")
        print("Mean:\n", self.model.mean)
        print("Cov:\n", self.model.cov)


class DeepGaussianBayesCritic(GaussianBayesCritic):
    """
    Bayesian linear model using a gaussian prior with known variance.

    Samples both the Q- and target Q-value by sampling the parameters per episode.
    """

    def __init__(self, state, lr=0.01):
        """
        Initializes a bayesian linear model.

        args:
            state : State from the environment of interest.
            lr    : Learning rate used by the linear model. 
        """
        super().__init__(state, lr)
        self.coef = self.sample_coef()

    def get_target_action_and_q_value(self, state):
        """Samples an action by picking the largest Q-value."""
        Q_left = self.q_value(state, 0)
        Q_right = self.q_value(state, 1)
        if Q_left > Q_right:
            return 0, Q_left
        return 1, Q_right

    def reset(self):
        """Samples coefficients on episode reset."""
        self.coef = self.sample_coef()
