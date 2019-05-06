from src.agents.util import featurizer
from numpy import array
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'done'])

def calculate_target(episode, transitions, gamma, next_q_value):
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
    n_step = len(transitions)

    discounted_rewards = [transition.reward*gamma**i for i, transition in enumerate(transitions)]
    reward_sum = sum(discounted_rewards)

    if not transitions[-1].done:
        target = reward_sum + gamma**n_step*next_q_value
        target = array([target]).reshape((1,))
        return target

    target = [reward_sum]
    for i in range(1,n_step):
        discounted_rewards = [transition.reward*gamma**n for n, transition in enumerate(transitions[-i:])]
        target.append(sum(discounted_rewards))
        
    target = array([target]).reshape((-1,1))
    return target


def q_learner(env, Critic, episodes=10000, gamma=1, verbose=False):
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
    n_step = 5
    steps = 0
    transitions = []

    for episode in range(1, episodes+1):


        state = env.reset()
        critic.reset()

        action = critic.get_action(state)
        done = False


        while not done:

            # Perform step
            next_state, reward, done, _ = env.step(action)
            transitions += [Transition(state, action, reward, done)]

            # Best next action
            _, next_q_value = critic.get_target_action_and_q_value(
                next_state)

            # Update parameters
            if steps > n_step:
                target = calculate_target(episode, transitions[-n_step:], gamma, next_q_value)
            
                if len(target) == 1:
                    critic.update(transitions[-n_step].state, transitions[-n_step].action, target)
                else:
                    for i in range(len(target)):
                        critic.update(transitions[-(n_step-i)].state, transitions[-(n_step-i)].action, target[i])

            # Reset loop
            state = next_state
            action = critic.get_action(state)
            steps += 1

        average_regret -= average_regret / 20
        average_regret += (1 - reward) / 20

        if average_regret < 0.01*1:  # What should "learned" be?
            break  # Check that this does not remove episode

    if verbose:
        print("Final Parameters")
        critic.print_parameters()
        critic.print_policy(num_states=env.N)
        print("Episodes used: ", episode)

    return episode
