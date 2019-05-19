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
        self.cov = np.eye(dim)*1e-3
        self.noise = 1e-12
        self.dim = dim

    def update_posterior(self, X, y, n):
        y = y.reshape((n, 1))

        inv_cov = np.linalg.inv(self.cov)

        self.mean = np.linalg.inv(X.T@X + inv_cov) @ \
            (X.T@y + inv_cov@self.mean)
        self.cov = np.linalg.inv(X.T @ X + inv_cov)

    def sample(self, X):
        beta_sample = self.sample_params()
        return self.sample_y(X, beta_sample)

    def sample_params(self):
        coef = np.random.multivariate_normal(
            self.mean[:, 0], self.cov)
        return coef

    def sample_y(self, X, beta_sample):
        return stats.norm.rvs(X@beta_sample.reshape(-1,1), np.sqrt(self.noise))

    def pdf(self, x, X):
        return np.squeeze(stats.norm.pdf(x, X@self.mean, (np.eye(self.dim)*self.noise)))

    def print_parameters(self):
        print("Mean\n", self.mean)
        print("Cov\n", self.cov)


class GaussianRegression2():
    def __init__(self, dim=3):
        self.mean = np.zeros((dim, 1))
        self.invcov = np.eye(dim)
        self.cov = np.linalg.inv(self.invcov)
        self.a = np.ones((20,20))*1
        self.b = np.ones((20,20))*1e-3
        self.dim = dim
        self.counter = 0

    def reset_var_params(self):
        self.counter += 1
        if self.counter==100:
            self.mean = np.zeros((self.dim, 1))
            
            self.invcov = np.eye(self.dim)
            self.cov = np.linalg.inv(self.invcov)
            self.a = np.ones((20,20))*1
            self.b = np.ones((20,20))*1e-3
            self.counter = 0

    def update_posterior(self, X, y, n):
        
        y = y.reshape((n, 1))
        
        mean_0 = self.mean
        invcov_0 = self.invcov
        
        self.invcov = X.T@X + self.invcov
        self.cov = np.linalg.inv(self.invcov)
        self.mean = self.cov@(X.T@y + invcov_0@mean_0)

        # step = int(X[0,0]-1)
        # state = int(X[0,1]-1)

        step=0
        state=0

        self.a[step,state] += n/2
        
        self.b[step,state] += max(0.5*np.asscalar(y.T@y -
            self.mean.T@(X.T@X + invcov_0)@self.mean + mean_0.T@invcov_0@mean_0), 1e-12)

    def sample(self, X, normal_vector):
        # step = int(X[0,0]-1)
        # state = int(X[0,1]-1)
        step=0
        state=0

        sigma_2 = stats.invgamma.rvs(self.a[step, state], scale=self.b[step,state])
        beta_sample = self.mean[:,0] + np.linalg.cholesky(self.cov)@normal_vector*np.sqrt(sigma_2)

        return self.sample_y(X, beta_sample, sigma_2)


    def sample_y(self, X, beta_sample, sigma_2):
        return stats.norm.rvs(X@beta_sample.reshape(-1,1), np.sqrt(sigma_2))

    def pdf(self, x, X):
        return multivariate_student_t(
            x, X@self.mean, self.b/self.a*(np.eye(self.dim)+X@self.cov@X.T), 2*self.a)


    def print_parameters(self):
        print("Mean\n", self.mean)
        print("Cov\n", self.cov)
        print("Gamma shape", self.a)
        print("Gamma scale", self.b)

