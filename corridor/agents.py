from models import MoveRight
from numpy.random import binomial

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

def q_learner(env, critic, episodes=10000):
    for episode in range(0,episodes):

        state = env.reset()
        action = critic.best_action(state)
        done = False
        steps = 0

        while not done:

            if binomial(1,0.1):
                 next_state, reward, done, _ = env.step(binomial(1, 0.5))
            else:
                # Perform step
                next_state, reward, done, _ = env.step(action)

            # Calculate Q-values
            q_value = critic.q_value(state, action)            

            # Best next action
            next_action = critic.best_action(next_state)
            next_q_value = critic.q_value(next_state, next_action)

            # Update parameters
            critic.update(reward, q_value, next_q_value, done)

            # Reset loop
            state = next_state
            action = next_action
            steps += 1

        print("Steps taken: ", steps)

    print("Final Parameters")
    critic.print_parameters()
    critic.print_q_values(num_states=env.N)
        
def move_right(env):

    move_right = MoveRight()

    _ = env.reset()
    action = move_right.policy()
    done = False

    steps = 0

    while not done:

        # Perform step
        _, _, done, _ = env.step(action)

        # Calculate Q-values
        action = move_right.policy()

        steps += 1

    print("Steps taken: ", steps)
