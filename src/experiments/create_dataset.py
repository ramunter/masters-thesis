from numpy import arange

from src.envs.corridor import Corridor
from src.envs.deepsea import DeepSea

from src.agents.q_learner_dataset_creator import q_learner
from src.agents.q_learner_critics import *


env = Corridor(N=5)
steps_to_learn = q_learner(
    env, GaussianBayesCritic2, episodes=1000, verbose=False)
print(steps_to_learn)
