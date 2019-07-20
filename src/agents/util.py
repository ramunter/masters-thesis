import numpy as np
import scipy.stats as stats

import numpy as np
from scipy.special import gammaln
from numpy.linalg import slogdet
from sklearn.linear_model import SGDRegressor

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
        self.XTX = np.zeros((dim,dim))
        self.XTy = np.zeros((dim,1))
    def update_posterior(self, X, y, n):

        y = y.reshape((n, 1))
    

        mean_0 = np.zeros((self.dim, 1))
        invcov_0 = np.linalg.inv(np.eye(self.dim)*1e-3)

        lr = 1-1e-3
        self.XTX = lr*self.XTX + X.T@X
        self.XTy = lr*self.XTy + X.T@y

        self.invcov = self.XTX + invcov_0
        self.cov = np.linalg.inv(self.invcov)
        self.mean = self.cov@(self.XTy + invcov_0@mean_0)

        # y = y.reshape((n, 1))

        # inv_cov = np.linalg.inv(self.cov)

        # self.mean = np.linalg.inv(X.T@X + inv_cov) @ \
        #     (X.T@y + inv_cov@self.mean)
        # self.cov = np.linalg.inv(X.T @ X + inv_cov)

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

        self.a = 1 + 1e-3
        self.dim = dim
        self.counter = 0

        self.XTX = np.zeros((dim,dim))
        self.XTy = np.zeros((dim,1))
        self.yTy = 0
        self.n = 0
        self.var = 0

        self.var = SGDRegressor(learning_rate="constant", eta0=0.01)
        self.var.partial_fit(np.ones((1,dim)), np.array([0]))
        self.var.coef_ = np.array([1.]*(dim))
        self.var.intercept_ = np.array([1.])
        self.var.t_ = 2

    def var_prediction(self, X):
        return self.var.predict(X.reshape(1,-1))

    def expected_variance(self,X):
        return self.beta(X)/(self.a-1)

    def beta(self, X):
        return abs(self.var.predict(X.reshape(1,-1))*self.a)

    def update_posterior(self, X, y, ey, td, var, n):

        y = y.reshape((n, 1))
        ey = ey.reshape((n, 1))

    
        lr = 1-1e-2

        mean_0 = np.zeros((self.dim, 1))
        invcov_0 = np.eye(self.dim)*1e-3

        a_0 = 1 + 1e-3

        self.XTX = lr*self.XTX + X.T@X
        self.XTy = lr*self.XTy + X.T@ey
        self.n = lr*self.n + n
        self.a = a_0 + self.n/2

        self.invcov = self.XTX + invcov_0
        self.cov = np.linalg.inv(self.invcov)
        self.mean = self.cov@(self.XTy + invcov_0@mean_0)

        # self.yTy = lr*self.yTy + self.mean.T@X.T@X@self.mean 
        # self.b = abs(b_0 + 0.5*np.asscalar(self.yTy - 
        #     self.mean.T@self.invcov@self.mean))
        tar_var = var + (2*ey-td)*td
        self.var.partial_fit(X.reshape(1,-1), np.array([tar_var]).reshape(1,))
        # self.b = b_0 + 0.5*(self.var)

    def sample(self, X, normal_vector):
        sigma_2 = stats.invgamma.rvs(self.a, scale=self.beta(X))
        beta_sample = self.mean[:,0] + np.linalg.cholesky(self.cov)@normal_vector*np.sqrt(sigma_2)
        sample = self.sample_y(X, beta_sample, sigma_2)
        return sample

    def sample_e(self, X, normal_vector):
        sigma_2 = stats.invgamma.rvs(self.a, scale=self.beta(X))
        beta_sample = self.mean[:,0] + np.linalg.cholesky(self.cov)@normal_vector*np.sqrt(sigma_2)
        return self.sample_ey(X, beta_sample)

    def sample_y(self, X, beta_sample, sigma_2):
        return stats.norm.rvs(X@beta_sample.reshape(-1,1),  sigma_2)

    def sample_ey(self, X, beta_sample):
        return X@beta_sample.reshape(-1,1)

    # def pdf(self, x, X):
    #     return multivariate_student_t(
    #         x, X@self.mean, self.beta(X)/self.a*(np.eye(self.dim)+X@self.cov@X.T), 2*self.a)


    def print_parameters(self):
        print("Mean\n", self.mean)
        print("Cov\n", self.cov)
        print("Gamma shape", self.a)
        print("var params", self.var.coef_)
        # print("Q_value 1,1", np.array([[1,1,1]])@self.mean)

