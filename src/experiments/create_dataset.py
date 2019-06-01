from numpy import arange

from src.envs.corridor import Corridor
from src.envs.deepsea import DeepSea

from src.agents.q_learner_dataset_creator import q_learner
from src.agents.q_learner_critics import *

env = Corridor(N=8)
steps_to_learn = q_learner(
    env, DeepGaussianBayesCritic2, episodes=3000, verbose=True)

print(steps_to_learn)
