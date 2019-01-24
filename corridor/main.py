from agents import actor_critic, q_learner, optimal
from models import Actor, Critic
from corridor import Corridor

env = Corridor(N=5, K=0)
optimal(env)

critic = Critic()
q_learner(env, critic, episodes=100)

