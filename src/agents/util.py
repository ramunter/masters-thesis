import numpy as np
import scipy.stats as stats

import numpy as np
from scipy.special import gammaln
from numpy.linalg import slogdet

def multivariate_student_t(X, mu, Sigma, df):    
    #multivariate student T distribution
    d = 1
    n = 1
    Xm = X-mu
    (_, logdet) = slogdet(df*Sigma)
    logz =  gammaln(0.5*(df + d)) - gammaln(df/2.0) - 0.5*d*np.log(np.pi) - 0.5*logdet 
    logp = -(df + d)/2 * np.log(1 + 1/df*Xm.T@np.linalg.inv(Sigma)@Xm)

    logp = logp + logz

    return np.exp(logp)

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
        self.cov = np.eye(dim)
        self.noise = noise
        self.dim = dim

    def update_posterior(self, X, y, n):
        y = y.reshape((n, 1))

        inv_cov = np.linalg.inv(self.cov)
        self.cov = np.linalg.inv(
            self.noise**(-2)*X.T@X + inv_cov)
        self.mean = self.cov @ \
            (X.T@y + inv_cov@self.mean)


    def sample(self, X):
        beta_sample = self.sample_params()
        return self.sample_y(X, beta_sample)

    def sample_params(self):
        coef = stats.multivariate_normal.rvs(
            self.mean[:,0],self.cov)
        return coef

    def sample_y(self, X, beta_sample):
        return X@stats.multivariate_normal.rvs(beta_sample, self.cov)


    def pdf(self, x, X):
        return np.squeeze(stats.norm.pdf(x, X@self.mean, (np.eye(self.dim)+X@self.cov@X.T)*self.noise))

    def print_parameters(self):
        print("Mean\n", self.mean)
        print("Cov\n", self.cov)


class GaussianRegression2():
    def __init__(self, dim=3):
        self.mean = np.zeros((dim, 1))
        self.invcov = np.eye(dim)*1e-3
        self.cov = np.linalg.inv(self.invcov)
        self.a = 1 + 1e-6
        self.b = 1e-6
        self.dim = dim

    def update_posterior(self, X, y, n):

        y = y.reshape((n, 1))

        mean_0 = self.mean
        invcov_0 = self.invcov
        w = np.diag([1]*n)# +  np.diag([1]*(n-1), k=1) #+ np.diag([1]*(n-1), k=-1)
        inv_w = np.linalg.inv(w)

        self.invcov += X.T@X
        self.cov = np.linalg.inv(self.invcov)

        self.mean = self.cov@(X.T@y + invcov_0@mean_0)

        self.a = self.a + n/2
        
        self.b = self.b + 0.5*np.asscalar(y.T@y + mean_0.T@invcov_0@mean_0 -
            self.mean.T@self.invcov@self.mean)

    def sample(self, X):
        beta_sample, sigma_2 = self.sample_params()
        return self.sample_y(X, beta_sample, sigma_2)

    def sample_params(self):
        sigma_2 = stats.invgamma.rvs(self.a, scale=self.b)
        beta_sample = stats.multivariate_normal.rvs(
            self.mean[:,0], self.cov*sigma_2)
        return beta_sample, sigma_2

    def sample_y(self, X, beta_sample, sigma_2):
        return stats.norm.rvs(X@beta_sample.reshape(-1,1), np.sqrt(sigma_2))

    def pdf(self, x, X):
        return multivariate_student_t(
            x, X@self.mean, self.b/self.a*(np.eye(self.dim)+X@self.cov@X.T), 2*self.a)

        # return stats.t.pdf(x, 2*self.a, X@self.mean, self.b/self.a*(np.eye(self.dim)+X@self.cov@X.T)).reshape(-1)

    def print_parameters(self):
        print("Mean\n", self.mean)
        print("Inv Cov\n", self.invcov)
        print("Gamma shape", self.a)
        print("Gamma scale", self.b)
class BayesTestRegression():
    def __init__(self, dim=3):
        self.mean = np.zeros((dim, 1))
        self.cov = np.eye(dim)
        self.invcov = np.linalg.inv(self.cov)
        self.noise = 0.5

        self.dim = dim

    def update_posterior(self, X, y, n):
        y = y.reshape((n, 1))
        w = np.diag([1]*n) +  np.diag([1]*(n-1), k=1) #+ np.diag([1]*(n-1), k=-1)
        inv_w = np.linalg.inv(w)
        invcov_0 = self.invcov
        self.invcov += X.T@inv_w@X
        self.cov = np.linalg.inv(self.invcov)

        self.mean = self.cov@(invcov_0@self.mean + X.T@inv_w@y)

    def sample(self, X):
        beta_sample = self.sample_params()
        return self.sample_y(X, beta_sample)

    def sample_params(self):
        coef = stats.multivariate_normal.rvs(
            self.mean[:,0],self.cov)
        return coef

    def sample_y(self, X, beta_sample):
        return X@stats.multivariate_normal.rvs(beta_sample, self.cov)

    def print_parameters(self):
        print("Mean\n", self.mean)
        print("Cov\n", self.cov)

    def pdf(self, x, X):
        return np.squeeze(stats.norm.pdf(x, X@self.mean, (np.eye(self.dim)+X@self.cov@X.T)*self.noise))
