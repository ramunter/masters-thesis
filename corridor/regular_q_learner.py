from numpy.random import binomial
from numpy import array
from sklearn.linear_model import SGDRegressor

from util import featurizer


def q_learner(env, episodes=10000):
    
    critic = Critic()

    state = env.reset()
    critic.init_model(state)

    for episode in range(0,episodes):

        state = env.reset()
        action = critic.get_action(state)
        done = False
        steps = 0

        while not done:

            # Perform step
            next_state, reward, done, _ = env.step(action)

            # Calculate Q-values
            q_value = critic.q_value(state, action)            

            # Best next action
            next_action = critic.best_action(next_state)
            next_q_value = critic.q_value(next_state, next_action)

            # Update parameters
            critic.update(state, action, reward, next_q_value, done)

            # Reset loop
            state = next_state
            action = critic.get_action(state)
            steps += 1

    # print("Final Parameters")
    # critic.print_parameters()
    # critic.print_policy(num_states=env.N)

    return critic.is_policy_optimal(env.N)

class Critic():

    def __init__(self, lr=0.01, gamma=0.9):

        self.model = SGDRegressor(learning_rate="constant", eta0=lr)
        self.gamma = gamma
    
    def init_model(self, state):
        features = featurizer(state, 0)
        self.model.partial_fit(features, array([0])) # SKlearn needs partial fit to be run once before use

    def q_value(self, state, action):
        features = featurizer(state, action)
        return self.model.predict(features)[0]

    def get_action(self, state):

        if binomial(1,0.2):
            return binomial(1, 0.5)
        else:
            # Perform step
            return self.best_action(state)

    def best_action(self, state):
        left_value = self.q_value(state, 0)
        right_value = self.q_value(state, 1)
        if left_value > right_value:
            return 0
        return 1

    def update(self, state, action, reward, next_q_value, done):
        target = reward
        if not done:
            target += self.gamma*next_q_value

        features = featurizer(state, action)
        target = array([target]).reshape((1,)) # Correct dim for SKlearn        

        self.model.partial_fit(features, target)

    def is_policy_optimal(self, num_states):
        # Moving right is optimal. Checks if all actions are right.
        policy = [self.best_action(state)==1 for state in range(0,num_states)]
        return all(policy)


    def print_parameters(self):
        print('Coefficients: \n', self.model.coef_)

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