#WIP

from models import MoveRight
from numpy.random import binomial
from numpy import array

def actor_critic(env, actor, critic, episodes=10000):    
    for episode in range(0,episodes):
        state = env.reset()
        action = actor.policy(state)
        done = False
        
        steps = 0

        while not done:
            
            # Perform step
            next_state, reward, done, _ = env.step(action)            
            next_action = actor.policy(next_state)

            # Calculate Q-values
            q_value = critic.q_value(state, action)
            next_q_value = critic.q_value(next_state, next_action)

            # Update parameters
            actor.update(q_value, action)
            critic.update(reward, q_value, next_q_value)

            # Reset loop
            state = next_state
            action = next_action
            steps += 1

        print("Steps taken: ", steps)