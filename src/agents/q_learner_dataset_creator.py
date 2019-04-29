from src.agents.util import featurizer
from numpy import array
import numpy as np
import pandas as pd


def calculate_target(gamma, reward, next_q_value, done):
    """
    Caclulates the target Q-value using temporal differencing.

    The Q-learning temporal target is Q(s,a) = r + gamma*Q(s', a') if in a 
    non-terminal state. Otherwise Q(s,a) = r  

    args:  
        gamma (float): The decay factor.  
        reward (int): Reward from action a.  
        next_q_value (float): Equivalent to Q(s', a')  
        done (bool): Is the current state a terminal state_ 

    returns:
        target (float): The Q-value target.
    """
    target = reward
    if not done:
        target += gamma*next_q_value
    target = array([target]).reshape((1,))
    return target


def q_learner(env, Critic, episodes=10000, gamma=0.9, verbose=False):
    """
    Runs a Q-learning experiment using the given environment and agent.

    This experiment must use a critic agent. The experiment is run until a
    running average of the regret falls below a given threshold or the maximum
    number of episodes is reached. Note that this function assumes the optimal
    reward is 1.

    args:  
        env (gym.Environment): The environment to test on.  
        Critic (CriticTemplate): The class of the agent to use. Has to be a critic.  
        episodes (int): Maximum number of episodes to run.  
        verbose (bool): Should this print information about the final model?  

    returns:  
        episode (int): Number of episodes run.
    """

    state = env.reset()
    critic = Critic(state)

    average_regret = 1

    dataset = []

    for episode in range(1, episodes+1):

        state = env.reset()
        critic.reset()

        action = critic.get_action(state)
        done = False
        steps = 0

        while not done:

            # Perform step
            next_state, reward, done, _ = env.step(action)

            # Best next action
            _, next_q_value = critic.get_target_action_and_q_value(
                next_state)

            # Update parameters
            target = calculate_target(gamma, reward, next_q_value, done)
            X = critic.update(state, action, target)

            dataset.append(
                np.append(X, [target[0], critic.q_value(state, action)]))

            # Reset loop
            state = next_state
            action = critic.get_action(state)
            steps += 1

        average_regret -= average_regret / 20
        average_regret += (1 - reward) / 20

    if verbose:
        print("Final Parameters")
        critic.print_parameters()
        critic.print_policy(num_states=env.N)
        print("Episodes used: ", episode)

    dataset = pd.DataFrame(dataset)
    print(dataset)
    print("Average regret is ", average_regret)
    dataset.to_csv("test.csv")

    return episode
