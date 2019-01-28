from absl import flags

from numpy import arange

from experiment import increasing_chain_length_experiment

from regular_q_learner import q_learner
from ube_q_learner import ube_q_learner, sample_target_ube_q_learner


flags.DEFINE_integer("iterations", 5, "Number of attempts per chain length")
flags.DEFINE_integer("longest_chain", 10, "Longest chain attempted")

FLAGS = flags.FLAGS 


increasing_chain_length_experiment(
    [q_learner, ube_q_learner, sample_target_ube_q_learner],
    ["Regular", "UBE", "UBE with sample target"],
    chain_length_sequence=arange(2, FLAGS.longest_chain, 2),
    attempts_per_chain_length=FLAGS.iterations)

