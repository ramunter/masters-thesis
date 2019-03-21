"""
I'm unsure how to test these properly. Most of these tests just check the
code doesn't crash to check that each method runs without crashing
"""

import unittest

import numpy as np

from src.agents.q_learner_critics import *


class TestCritics(unittest.TestCase):

    def setUp(self):
        self.critics = [EGreedyCritic, UBECritic, SampleTargetUBECritic,
                        GaussianBayesCritic, DeepGaussianBayesCritic]

    def all_template_functions_run(self, critic_class, state):
        """
        Checks that all template functions are working correclty for a given
        critic.
        """
        critic = critic_class(state)

        action = critic.get_action(state)

        self.assertIsInstance(action, int)

        action, q_value = critic.get_target_action_and_q_value(state)

        self.assertIsInstance(action, int)
        self.assertIsInstance(q_value, float)

        # Just check that these don't crash
        critic.update(state, 0, np.array([0]))
        action = critic.best_action(state)

        # Check batch update
        critic.batch = True
        critic.update([state]*10, [0]*10, np.array([0]*10))

    def test_all_critics_run_with_int_state(self):
        """
        Tests that all critics are runnable with an int state.
        """
        state = 0
        for critic in self.critics:
            self.all_template_functions_run(critic, state)

    def test_all_critics_run_with_list_state(self):
        """
        Tests that all critics are runnable with a list state.
        """
        state = [0, 0]
        for critic in self.critics:
            self.all_template_functions_run(critic, state)

    def test_all_critics_run_with_np_state(self):
        """
        Tests that all critics are runnable with a numpy state.
        """
        state = np.array([0, 0, 0])
        for critic in self.critics:
            self.all_template_functions_run(critic, state)


if __name__ == '__main__':
    unittest.main()
