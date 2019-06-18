from abc import ABC, abstractmethod
from collections import namedtuple, deque

import numpy as np
from numpy.random import binomial
import scipy.stats as stats

from sklearn.linear_model import SGDRegressor, SGDClassifier
from filterpy.kalman import KalmanFilter

from src.agents.util import featurizer, GaussianRegression, GaussianRegression2


class CriticTemplate(ABC):
    """
    Template for a critic agent that can be used with the Q-learner
    experiment above.
    """

    # Functionality

    @abstractmethod
    def get_action(self, state):
        pass    
        
    @abstractmethod
    def get_eval_action(self, state):
        pass

    @abstractmethod
    def get_target_action_and_q_value(self, state):
        pass

    @abstractmethod
    def update(self, state, action, target):
        pass

    def reset(self):
        pass

    # Debug functions

    @abstractmethod
    def print_parameters(self):
        pass

    def print_policy(self, num_states):
        print("Policy ", end='')
        for state in range(0, num_states):
            if self.best_action(state):
                print("r", end=' ')
            else:
                print("l", end=' ')
        print()


class EGreedyCritic(CriticTemplate):
    """ A regular E greedy agent with a constant E."""

    def __init__(self, state, batch=False, final_eps=1e-3, lr=0.01):
        """
        Initializes a linear model.

        args:
            state : State from the environment of interest.
            eps   : Probability of choosing a random action.
            batch : Batch or single updates?
            lr    : Learning rate used by the linear model.
        """
        self.final_eps = final_eps
        self.eps = 0.1
        self.eps_decay = (1-final_eps)/200
        self.batch = batch
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
        if binomial(1, self.eps):
            return binomial(1, 0.5)
        else:
            return self.get_eval_action(state)

    def get_eval_action(self, state):
        action, q_value = self.get_target_action_and_q_value(state)
        return action

    def get_target_action_and_q_value(self, state):
        """ Calculates the optimal action and Q-value."""

        left_value = self.q_value(state, 0)
        right_value = self.q_value(state, 1)
        if left_value > right_value:
            return 0, left_value
        return 1, right_value

    def update(self, state, action, target):
        """Takes one optimization step for the linear model."""

        X = featurizer(state, action, self.batch)
        self.model.partial_fit(X, np.array([target]).reshape(1,))
        return X

    def q_value(self, state, action):
        """Caclulates Q-value given state and action."""
        features = featurizer(state, action)
        return self.model.predict(features)[0]

    def print_parameters(self):
        print('Coefficients: \n', self.model.coef_)

    def reset(self):
        if self.eps > self.final_eps:
            self.eps -= self.eps_decay


class UBECritic(CriticTemplate):
    """ The uncertainty bellman equation method without propagating uncertainty"""

    def __init__(self, state, batch=False, lr=0.01):
        """
        Initializes a linear model.

        args:
            state : State from the environment of interest.
            lr    : Learning rate used by the linear model. 
            batch : Batch or single updates?
        """
        self.batch = batch
        self.model = self.setup_model(state, lr)
        self.sigma = np.eye(featurizer(state, 0, False).shape[1])
        self.beta = 6

    def setup_model(self, state, lr):
        model = SGDRegressor(learning_rate="constant",
                             eta0=lr, fit_intercept=False)
        features = featurizer(state, action=0)
        # SKlearn needs partial fit to be run once before use
        model.partial_fit(features, np.array([0]))
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
        prediction = self.model.predict(features)
        return np.asscalar(prediction)

    def action_variance(self, state, action):
        """Caclulate the Q-value variance based on the equation in the UBE paper."""
        features = featurizer(state, action)
        var_action = features@self.sigma@features.T
        return np.asscalar(var_action)

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
        return np.asscalar(sample_q)

    def update(self, state, action, target):
        """Train the model using Q-learning TD update."""
        features = featurizer(state, action, self.batch)
        self.model.partial_fit(features, target)

        num_samples = features.shape[0]

        for i in range(0, num_samples):
            self.update_sigma(features[i, :])

        return features

    def update_sigma(self, features):
        """Update the Covariance matrix."""

        change_numerator = self.sigma * features.T * \
            features * self.sigma

        change_denominator = 1 + features @ self.sigma @ features.T
        self.sigma -= change_numerator/change_denominator

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

    def __init__(self, state, batch=False, lr=0.01):
        """
        Initializes a bayesian linear model.

        args:
            state : State from the environment of interest.
            lr    : Learning rate used by the linear model. 
            batch : Batch or single updates?
        """
        self.count = 0

        self.batch = batch
        if type(state) is int:
            feature_size = 2
        else:
            feature_size = len(state) + 1  # Add bias term.

        self.models = [GaussianRegression(dim=feature_size), GaussianRegression(dim=feature_size)] # Model per action
        self.final_eps = 0.01
        self.eps = 1
        self.eps_decay = (1-self.final_eps)/200

    def get_action(self, state):
        Q_left = self.q_value(state, 0)
        Q_right = self.q_value(state, 1)
        if Q_left > Q_right:
            return 0
        return 1

    def get_eval_action(self, state):
        Q_left = self.mean_q_value(state, 0)
        Q_right = self.mean_q_value(state, 1)
        if Q_left > Q_right:
            return 0
        return 1

    def get_target_action_and_q_value(self, state):
        """
        Samples an action by sampling coefficients and choosing the highest
        resulting Q-value.
        """
        Q_left = self.q_value(state, 0)
        Q_right = self.q_value(state, 1)
        if Q_left > Q_right:
            return 0, Q_left
        return 1, Q_right

    def update(self, state, action, target):
        """Calculate posterior and update prior."""
        X = self.featurizer(state)
        self.models[action].update_posterior(X, target, 1)
        X = np.array(np.append(state, [action, 1]))
        return X

    def q_value(self, state, action):
        """Caclulate Q-value based on sampled coefficients."""
        X = self.featurizer(state)
        prediction = self.models[action].sample(X)

        return np.asscalar(prediction)

    def mean_q_value(self, state, action):
        """Caclulate Q-value based on sampled coefficients."""
        X = self.featurizer(state)
        prediction = X@self.models[action].mean

        return np.asscalar(prediction)

    def featurizer(self, state):
        #return np.append([state, 1], self.policy.coef_).reshape(1,-1)
        return np.append(state, [1]).reshape(1,-1)

    def print_parameters(self):
        for action in range(2):
            print("Action ", action)
            self.models[action].print_parameters()

    def reset(self):
        if self.eps > self.final_eps:
            self.eps -= self.eps_decay


class DeepGaussianBayesCritic(GaussianBayesCritic):
    """
    Bayesian linear model using a gaussian prior with known variance.

    Samples both the Q- and target Q-value by sampling the parameters per episode.
    """

    def __init__(self, state, batch=False, lr=0.01):
        """
        Initializes a bayesian linear model.

        args:
            state : State from the environment of interest.
            lr    : Learning rate used by the linear model. 
            batch : Batch or single updates?
        """
        super().__init__(state, batch, lr)
        self.coef = [model.sample_params() for model in self.models]

    def get_target_action_and_q_value(self, state):
        """Samples an action by picking the largest Q-value."""
        Q_left = self.q_value(state, 0)
        Q_right = self.q_value(state, 1)
        if Q_left > Q_right:
            return 0, Q_left
        return 1, Q_right

    def q_value(self, state, action):
        """Caclulate Q-value based on sampled coefficients."""
        features = self.featurizer(state)
        prediction = self.models[action].sample_y(features, self.coef[action])
        return np.asscalar(prediction)

    def reset(self):
        """Samples coefficients on episode reset."""
        self.coef = [model.sample_params() for model in self.models]


class DeepGaussianBayesCritic2(CriticTemplate):
    """
    Bayesian linear model using a gaussian prior with unknown variance.

    Samples both the Q- and target Q-value by sampling the parameters per step.
    """

    def __init__(self, state, batch=False, lr=0.01):
        """
        Initializes a bayesian linear model.

        args:
            state : State from the environment of interest.
            lr    : Learning rate used by the linear model. 
            batch : Batch or single updates?
        """
        self.count = 0

        self.batch = batch
        if type(state) is int:
            feature_size = 2
        else:
            feature_size = len(state) + 1  # Add bias term.

        self.feature_size = feature_size
        self.models = [GaussianRegression2(dim=feature_size), GaussianRegression2(dim=feature_size)] # Model per action
        self.normal_samples = self.sample_standard_normal_vector()


    def get_action(self, state):
        Q_left = self.q_value(state, 0)
        Q_right = self.q_value(state, 1)
        if Q_left > Q_right:
            return 0
        return 1

    def get_eval_action(self, state):
        Q_left = self.mean_q_value(state, 0)
        Q_right = self.mean_q_value(state, 1)
        if Q_left > Q_right:
            return 0
        return 1

    def get_target_action_and_q_value(self, state):
        """
        Samples an action by sampling coefficients and choosing the highest
        resulting Q-value.
        """
        Q_left = self.q_value(state,  0, stats.norm.rvs(size=self.models[0].dim))
        Q_right = self.q_value(state, 1, stats.norm.rvs(size=self.models[1].dim))
        if Q_left > Q_right:
            return 0, Q_left
        return 1, Q_right

    def update(self, state, action, target):
        """Calculate posterior and update prior."""
        X = self.featurizer(state)
        self.models[action].update_posterior(X, target, 1)

    def q_value(self, state, action, normal_samples=None):
        """Caclulate Q-value based on sampled coefficients."""
        if normal_samples is None:
            normal_samples = self.normal_samples[action]

        X = self.featurizer(state)
        prediction = self.models[action].sample(X, normal_samples)
        return np.asscalar(prediction)

    def eq_value(self, state, action, normal_samples=None):
        """Caclulate Q-value based on sampled coefficients."""
        if normal_samples is None:
            normal_samples = self.normal_samples[action]

        X = self.featurizer(state)
        prediction = self.models[action].sample_e(X, normal_samples)
        return np.asscalar(prediction)

    def mean_q_value(self, state, action, normal_samples=None):
        """Caclulate Q-value based on sampled coefficients."""
        X = self.featurizer(state)
        prediction = X@self.models[action].mean
        return np.asscalar(prediction)

    def featurizer(self, state):
        return np.append(state, [1]).reshape(1,-1)

    def print_parameters(self):
        for action in range(2):
            print("Action ", action)
            self.models[action].print_parameters()

    def sample_standard_normal_vector(self):
        return [stats.norm.rvs(size=model.dim) for model in self.models]

    def reset(self):
        self.normal_samples = self.sample_standard_normal_vector()
        # [model.reset_var_params() for model in self.models]



class GaussianBayesCritic2(DeepGaussianBayesCritic2):
    def get_action(self, state):
        self.normal_samples = self.sample_standard_normal_vector()
        Q_left = self.q_value(state, 0)
        Q_right = self.q_value(state, 1)
        if Q_left > Q_right:
            return 0
        return 1

class KalmanFilterCritic(CriticTemplate):
    """
    Kalman filtered regression.
    """

    def __init__(self, state, batch=False, lr=0.01):
        """
        Initializes a bayesian linear model.

        args:
            state : State from the environment of interest.
            lr    : Learning rate used by the linear model. 
            batch : Batch or single updates?
        """
        self.eps = 1
        self.final_eps = 0.01
        self.eps_decay = (1-self.final_eps)/200
        self.batch = batch
        if type(state) is int:
            feature_size = 3
        else:
            feature_size = len(state) + 2  # Add bias term and action term.

        self.model = KalmanFilter(dim_x=feature_size, dim_z=1)
        self.model.x = np.zeros((feature_size,))
        self.model.F = np.eye(feature_size)  # x = Fx
        self.model.H = None  # y = Hx

    def get_action(self, state):
        """ Gets an action using the E greedy approach."""

        if binomial(1, self.eps):
            return binomial(1, 0.5)
        else:
            action, _ = self.get_target_action_and_q_value(state)
        return action

    def get_target_action_and_q_value(self, state):
        """
        Samples an action by sampling coefficients and choosing the highest
        resulting Q-value.
        """
        Q_left = self.q_value(state, 0)
        Q_right = self.q_value(state, 1)
        if Q_left > Q_right:
            return 0, Q_left
        return 1, Q_right

    def update(self, state, action, target):
        """Calculate posterior and update prior."""
        X = featurizer(state, action, self.batch)
        self.model.update(target, H=X)
        return X

    def q_value(self, state, action):
        """Caclulate Q-value based on sampled coefficients."""
        features = featurizer(state, action)
        prediction = features@self.model.x_post
        return np.asscalar(prediction)

    def print_parameters(self):
        print("Coefficients")
        print(self.model.predict())

    def reset(self):
        if self.eps > self.final_eps:
            self.eps -= self.eps_decay
