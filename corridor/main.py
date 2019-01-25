from agents import actor_critic, q_learner, move_right
from models import Actor, Critic
from corridor import Corridor

num_states = 20

env = Corridor(N=num_states, K=0)
move_right(env)

critic = Critic(lr=0.01)
q_learner(env, critic, episodes=10000)

