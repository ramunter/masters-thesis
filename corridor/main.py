from experiment import increasing_chain_length_experiment
from numpy import arange

from regular_q_learner import q_learner
from ube_q_learner import ube_q_learner, sample_target_ube_q_learner

increasing_chain_length_experiment(
    [q_learner, ube_q_learner, sample_target_ube_q_learner],
    ["Regular", "UBE", "UBE with sample target"],
    chain_length_sequence=arange(2, 20, 2),
    attempts_per_chain_length=5)

