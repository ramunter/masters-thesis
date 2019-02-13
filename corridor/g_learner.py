from abc import ABC, abstractmethod
from util import featurizer
from numpy import array

def calculate_target(gamma, reward, next_g_value, done):
    target = reward
    if not done:
        target += gamma*next_g_value
    target = array([target]).reshape((1,))
    return target

def g_learner(env, Critic, episodes=10000, gamma=1, verbose=False):
    
    state = env.reset()
    critic = Critic()

    critic.init_model(state)

    average_regret = 1

    for episode in range(1,episodes+1):

        state = env.reset()
        critic.reset()

        action = critic.get_action(state)
        done = False
        steps = 0

        states=[state]
        actions=[action]

        while not done:

            # Perform step
            state, reward, done, _ = env.step(action)

            # Reset loop
            action = critic.get_action(state)

            states += [state]
            actions += [action]
            steps += 1

        average_regret -= average_regret / 20
        average_regret += (1 - reward) / 20

        if average_regret < 0.01*1: # What should "learned" be? 
            break # Check that this does not remove episode

        for state, action in zip(states, actions):
            critic.update(state, action, [reward])

    if verbose:
        print("Final Parameters")
        critic.print_parameters()
        critic.print_policy(num_states=env.N)
        print("Episodes used: ", episode)

    return episode
