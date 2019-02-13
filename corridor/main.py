from absl import flags, app

from numpy import arange

from experiment import increasing_chain_length_experiment

from q_learner_critics import EGreedyCritic, UBECritic, SampleTargetUBECritic, GaussianBayesCritic, DeepGaussianBayesCritic

flags.DEFINE_integer("iterations", 5, "Number of attempts per chain length")
flags.DEFINE_integer("longest_chain", 10, "Longest chain attempted")

FLAGS = flags.FLAGS 

def main(argv):

    increasing_chain_length_experiment(
        [GaussianBayesCritic, DeepGaussianBayesCritic],
        ["Gaussian Prior", "Deep Gaussian Prior"],
        chain_length_sequence=arange(4, FLAGS.longest_chain, 2),
        attempts_per_chain_length=FLAGS.iterations)

if __name__ == '__main__':
    app.run(main)