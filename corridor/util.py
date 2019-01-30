from numpy import array

def featurizer(state, action):
    return array([state, action]).reshape(1,-1)

