
import unittest

import numpy as np

from src.agents.util import *


class TestUtil(unittest.TestCase):

    def test_featurizer(self):

        state = np.array([1, 2, 3, 4, 5]).reshape(5, 1)
        action = 0

        states = [state]*10
        actions = [action]*10

        feature = featurizer(state, action, batch=False)
        self.assertEqual(feature.shape[1], 7)

        features = featurizer(states, actions, batch=True)
        self.assertEqual(features.shape[1], 7)

    def test_gaussian_reg_2(self):

        dim = 2

        def verify_model_parameters(model):
            self.assertIsInstance(model.a, float)
            self.assertIsInstance(model.b, float)
            self.assertEqual(model.mean.shape, (dim, 1))
            self.assertEqual(model.invcov.shape, (dim, dim))

        model = GaussianRegression2(dim)
        verify_model_parameters(model)

        model.sample()
        verify_model_parameters(model)

        data = np.array([1, 2]).reshape(1, dim)
        model.update_posterior(data, np.array([1]), n=1)
        verify_model_parameters(model)


if __name__ == '__main__':
    unittest.main()
