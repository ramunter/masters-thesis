import numpy as np

def featurizer(state, action):
    """
    Creates an array containing the state, bias term and action.

    args:
        state (int, list or np.array): State.
        action (int)                 : Action.
    """
    return np.append(state, [1, action]).reshape(1,-1)

class GaussianRegression():
    def __init__(self, noise=1, dim=3):
        self.mean = np.zeros((dim,1))
        self.cov = np.eye(dim)
        self.noise = noise