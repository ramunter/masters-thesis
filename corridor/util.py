import numpy as np

def featurizer(state, action):
    return np.array([state, action]).reshape(1,-1)

class GaussianRegression():
    def __init__(self, noise=1, dim=3):
        self.mean = np.zeros((dim,1))
        self.cov = np.eye(dim)
        self.noise = noise