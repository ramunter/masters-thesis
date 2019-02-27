import unittest

import numpy as np

from src.envs.corridor import Corridor


class TestCorridor(unittest.TestCase):
    def setUp(self):
        self.corridor_0 = Corridor(N=2)
        self.corridor_1 = Corridor(N=5)
        self.corridor_2 = Corridor(N=100)

    def test_step_is_done_after_correct_number_of_steps(self):
        """
        Following an optimal policy we should only get a reward when
        transitioning to the terminal state.
        """
        def check_correct_rewards(env):
            done = False
            reward_list = []
            while not done:
                state = env.state

                optimal_action = 1
                state, reward, done, _ = env.step(
                    action=optimal_action)
                reward_list.append(reward)

            # Final reward is 1
            self.assertEqual(reward_list[-1], 1)
            # All rewards before this are 0
            self.assertEqual(sum(reward_list[0:-1]), 0)
            # Correct state
            self.assertEqual(env.state, env.N-1)
            # Correct number of steps
            self.assertEqual(env.steps, env.N)

        check_correct_rewards(self.corridor_0)
        check_correct_rewards(self.corridor_1)
        check_correct_rewards(self.corridor_2)
