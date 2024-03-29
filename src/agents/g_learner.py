from abc import ABC, abstractmethod
from src.agents.util import featurizer
from numpy import array


def g_learner(
        env,
        Critic,
        episodes=10000,
        gamma=1,
        verbose=False):
    """
    Runs a Q-learner experiment trained on the MC return using the
    given environment and agent.  

    This experiment must use a critic agent. The experiment is run until a
    running average of the regret falls below a given threshold or the maximum
    number of episodes is reached. Note that this function assumes the optimal
    reward is 1.  

    args:  
        env : The environment to test on.  
        Critic : The class of the agent to use. Has to be a critic.  
        episodes : Maximum number of episodes to run.  
        verbose : Should this print information about the final model?  

    returns:  
        episode : Number of episodes run.  
    """

    state = env.reset()
    critic = Critic(state, batch=True)

    average_regret = 1

    for episode in range(1, episodes+1):

        state = env.reset()
        critic.reset()

        action = critic.get_action(state)
        done = False
        steps = 0

        states = [state]
        actions = [action]

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

        critic.update(states, actions, [reward]*len(states))

        if average_regret < 0.01*1:  # What should "learned" be?
            break

    if verbose:
        print("Final Parameters")
        critic.print_parameters()
        critic.print_policy(num_states=env.N)
        print("Episodes used: ", episode)

    return episode
