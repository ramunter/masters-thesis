import unittest

import numpy as np

from src.envs.deepsea import DeepSea, State


class TestDeepSea(unittest.TestCase):
    def setUp(self):
        self.deepsea_0 = DeepSea(N=2)
        self.deepsea_1 = DeepSea(N=5)
        self.deepsea_2 = DeepSea(N=100)

    def test_action_state_handler(self):
        action = 0
        for x in range(0, self.deepsea_1.N):
            for y in range(0, self.deepsea_1.N):
                self.deepsea_1.state = State(x, y)
                self.assertTrue(self.deepsea_1.action_state_handler(action) ==
                                self.deepsea_1.is_reverse_state[x, y])

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

                optimal_action = 1 - \
                    env.is_reverse_state[state.x, state.y]

                state, reward, done, _ = env.step(
                    action=optimal_action)
                reward_list.append(reward)

            # Final reward is 1
            self.assertEqual(reward_list[-1], 1)
            # All rewards before this are 0
            self.assertEqual(sum(reward_list[0:-1]), 0)
            # Correct state
            self.assertEqual(env.state.x, env.N-1)
            self.assertEqual(env.state.y, env.N-1)
            # Correct number of steps
            self.assertEqual(env.steps, env.N)

        check_correct_rewards(self.deepsea_0)
        check_correct_rewards(self.deepsea_1)
        check_correct_rewards(self.deepsea_2)
