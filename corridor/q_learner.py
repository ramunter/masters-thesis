from abc import ABC, abstractmethod
from util import featurizer
from numpy import array

def calculate_target(gamma, reward, next_q_value, done):
    target = reward
    if not done:
        target += gamma*next_q_value
    target = array([target]).reshape((1,))
    return target

def q_learner(env, Critic, episodes=10000, gamma=1, verbose=False):
    
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

        if average_regret < 0.1*1: # What should "learned" be? 
            # print("Episode:", episode)
            # print(average_regret)
            # print(cumulative_regret)
            break # Check that this does not remove episode


    if verbose:
        print("Final Parameters")
        critic.print_parameters()
        critic.print_policy(num_states=env.N)

    return episode


class CriticTemplate(ABC):
    
    ## Functionality

    @abstractmethod
    def init_model(self, state):
        pass

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
