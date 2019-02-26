import unittest

import src.envs.deepsea import DeepSea, State

class TestDeepSea(unittest.TestCase):
    def setUp():
        deepsea_0 = DeepSea(N=1)
        deepsea_1 = DeepSea(N=5)
        deepsea_2 = DeepSea(N=100)

    def test_action_state_handler(self):
        action = 0
        for x in range(0, deepsea_10.N):
            for y in range(0, deepsea_10.N):
                deepsea_1.state = State(x,y)
                self.assertTrue(deepsea_1.action_state_handler(action) == \
                                deepsea_1.self.is_reverse_state[x,y])
