from abc import ABC, abstractmethod
from src.agents.util import featurizer
from numpy import array

def calculate_target(gamma, reward, next_q_value, done):
    """ Caclulates the target Q-value using temporal differencing.

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

    for episode in range(1,episodes+1):

        state = env.reset()
        critic.reset()

        action = critic.get_action(state)
        done = False
        steps = 0

        while not done:

            # Perform step
            next_state, reward, done, _ = env.step(action)

            # Best next action
            next_action, next_q_value = critic.get_target_action_and_q_value(next_state)

            # Update parameters
            target = calculate_target(gamma, reward, next_q_value, done)
            critic.update(state, action, target)

            # Reset loop
            state = next_state
            action = critic.get_action(state)
            steps += 1

        average_regret -= average_regret / 20
        average_regret += (1 - reward) / 20

        if average_regret < 0.01*1: # What should "learned" be? 
            break # Check that this does not remove episode


    if verbose:
        print("Final Parameters")
        critic.print_parameters()
        critic.print_policy(num_states=env.N)
        print("Episodes used: ", episode)

    return episode


class CriticTemplate(ABC):
    """Template for a critic agent that can be used with the Q-learner
    experiment above.
    """
    
    ## Functionality

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def get_target_action_and_q_value(self, state):
        pass

    @abstractmethod
    def update(self, state, action, reward, next_q_value, done):
        pass

    def reset(self):
        pass

    def best_action(self, state):
        action, q_value = self.get_target_action_and_q_value(state)
        return action

    def is_policy_optimal(self, num_states):
        policy = [self.best_action(state)==1 for state in range(0,num_states)]
        return all(policy)

    ## Debug functions
    
    @abstractmethod
    def print_parameters(self):
        pass

    def print_q_values(self, num_states):
        print("Left ", end='')
        for state in range(0,num_states):
            print(
                  round(self.q_value(state, 0), 2),
                  end=' ')
        print()
        
        print("Right ", end='')
        for state in range(0,num_states):
            print(
                  round(self.q_value(state, 1), 2),
                  end=' ')
        print()
    
    def print_policy(self, num_states):
        print("Policy ", end='')
        for state in range(0, num_states):
            if self.best_action(state):
                print("r", end=' ')
            else:
                print("l", end=' ')
        print()
