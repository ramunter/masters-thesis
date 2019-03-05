from numpy import arange

from src.envs.corridor import Corridor
from src.envs.deepsea import DeepSea

from src.agents.q_learner_dataset_creator import q_learner
from src.agents.q_learner_critics import EGreedyCritic, GaussianBayesCritic


env = DeepSea(N=4)
steps_to_learn = q_learner(
    env, GaussianBayesCritic, episodes=10000, verbose=False)
print(steps_to_learn)
