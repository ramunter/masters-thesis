import numpy as np
import scipy.stats as stats


def featurizer(state, action, batch=False):
    """
    Creates an array containing the state, bias term and action.

    args:
        state  : State.
        action : Action.
        batch  : Is this a batch update?
    """
    if batch:
        state = np.squeeze(state)
        bias = [1]*len(state)
        features = np.column_stack((state, action, bias))
    else:
        features = np.append(
            state, [action, 1]).reshape(1, -1)
    return features


class GaussianRegression():
    def __init__(self, noise=1, dim=3):
        self.mean = np.zeros((dim, 1))
        self.cov = np.eye(dim)*10
        self.noise = noise


class GaussianRegression2():
    def __init__(self, dim=3):
        self.mean = np.zeros((dim, 1))
        self.invcov = np.eye(dim)*1
        self.a = 1
        self.b = 1

    def update_posterior(self, X, y, n):

        y = y.reshape((n, 1))

        mean_0 = self.mean
        self.mean = np.linalg.inv(
            X.T@X + self.invcov)@(X.T@y + self.invcov@mean_0)

        invcov_0 = self.invcov
        self.invcov = X.T@X + self.invcov

        self.a += n/2
        self.b += 0.5*np.asscalar(y.T@y + mean_0.T@invcov_0@mean_0 -
                                  self.mean.T@self.invcov@self.mean)

    def sample(self):
        noise = stats.invgamma.rvs(self.a, scale=1./self.b)
        sample = stats.multivariate_normal.rvs(
            np.squeeze(self.mean), np.linalg.inv((self.invcov+self.invcov.T)/2.)*noise)
        return sample, noise

    def print_parameters(self):
        print("Mean\n", self.mean)
        print("Inv Cov\n", self.invcov)
        print("Gamma shape", self.a)
        print("Gamma scale", self.b)
