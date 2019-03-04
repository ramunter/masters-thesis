
import unittest

import numpy as np

from src.agents.util import *


class TestCritics(unittest.TestCase):

    def test_featurizer(self):

        state = np.array([1, 2, 3, 4, 5]).reshape(5, 1)
        action = 0

        states = [state]*10
        actions = [action]*10

        feature = featurizer(state, action, batch=False)
        self.assertEqual(feature.shape[1], 7)

        features = featurizer(states, actions, batch=True)
        self.assertEqual(features.shape[1], 7)


if __name__ == '__main__':
    unittest.main()
