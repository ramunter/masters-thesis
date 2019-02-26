from absl import flags, app
from numpy import arange

from src.experiments.experiment import experiment

from src.envs.corridor import Corridor
from src.envs.deepsea import DeepSea

from src.agents.q_learner import q_learner
from src.agents.g_learner import g_learner

from src.agents.q_learner_critics import EGreedyCritic, UBECritic, SampleTargetUBECritic, GaussianBayesCritic, DeepGaussianBayesCritic


flags.DEFINE_integer("iterations", 5, "Number of attempts per chain length")
flags.DEFINE_integer("longest_chain", 10, "Longest chain attempted")

FLAGS = flags.FLAGS 

def main(argv):
        methods = {"Q-learning": q_learner,
                   "G-learning": g_learner}

        critics = {"Gaussian Prior": GaussianBayesCritic,
                   "Deep Exploration Gaussian Prior": DeepGaussianBayesCritic}

        experiment(
                environment=DeepSea,#Corridor,
                N_list=arange(4, FLAGS.longest_chain, 2),
                methods=methods,
                critics=critics,
                attempts_per_N=FLAGS.iterations)

if __name__ == '__main__':
    app.run(main)