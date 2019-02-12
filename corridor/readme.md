# Documentation of Q-learning tests on a toy problem

## Corridor

This environment is a generalization of an environment called riverswim.

![Chain Env](./images/chain.png)

The corridor environment allows one to define:

* **N:** The environment consists of a N long corridor where the agent can move
  left or right.
* **K:** in a K number of states, the direction traveled when given is the
  opposite of the expected. I.e. action left will cause the agent to move right.
* **p:** Probability of success when moving right. 1-**p** probability of moving left
  instead.
        

## Experiment setup

The current setup gives 1 reward for reaching the final node to the right, and 0 otherwise. The agent has N steps to try to reach the goal. The experiment is run with K=0 and p=1 to allow a simple linear model and simple calculation of the optimal reward.

The agent is then tested in the environment multiple times for an increasing value of N. The attempt is stopped when the agent reaches a running average of regret per episode that is lower than 1% of the optimal reward. When this happens we consider that the agent has "learned" the environment.


### Experiment pseudocode

<pre>
<b>Increasing Chain Length Experiment</b>

<b>for</b> agent <b>in</b> agents
  <b>for</b> N <b>in</b> chain_lengths

    environment = corridor(N, 0, 1)

    <b>for</b> i <b>in</b> attempts
      steps_to_learn = q_learn(agent, environment)
    
    average_steps_to_learn = steps_to_learn/attempts
</pre>

## Agent

The agent is a vanilla Q-learning algorithm.

<pre>
<b>Q-learner</b>

<b>for</b> episode <b>in</b> episodes
  state, critic = reset environment and critic
  action = critic.get_action(state)

  <b>while</b> not done:

    next_state, reward, done = take_step(action)
    target = calculate Q-learning target
    critic.update(state, action, target)
    
    state = next_state
    action = critic.get_action(state)

  average_regret = running_average_regret()

  <b>if</b> average_regret < threshold:
    break
</pre>

## Critics

For each Critic pseudocode is provided for the key functionality of the critic.

### e-greedy

Uses a linear regression method using SGD for parameter optimization.

<pre>


<b>Get Action</b>

sample e from unif(0,1)
<b>if</b> e < 0.2:
  return random action
return best action


<b>Best Action</b>

return argmax Q-value(state, action | state)
</pre>

### Sample Target UBE Critic

Note that this implementation does not directly propagate the local uncertainty.


<pre>


<b>Get Action</b>

action = argmax Sample Q-value(state, action | state)


<b>Next Q-value and Action</b>
q_values = Sample Q-value(state, action | state)
next_action = argmax q_values
next_q_value = q_values[next_action]


<b>Sample Q-value</b>

mean_q = regression_prediction(state, action)
var_q = action_variance(state, action)
u = sample unif(0,1)
Q-value = mean_q + beta*var_q*u

</pre>

The action varaince calculation and update are best shown through their equations:

<!-- $
\\
\textbf{Action Variance}
\\\\
S^T\Sigma_aS
\\\\
\textbf{Update }\Sigma_a
\\\\
\Sigma_a = \Sigma_a - \frac{\Sigma_aSS^T\Sigma_a}{1 + S^T\Sigma_aS}
$ -->

![](https://latex.codecogs.com/gif.latex?%5C%5C%20%5Ctextbf%7BAction%20Variance%7D%20%5C%5C%5C%5C%20S%5ET%5CSigma_aS%20%5C%5C%5C%5C%20%5Ctextbf%7BUpdate%20%7D%5CSigma_a%20%5C%5C%5C%5C%20%5CSigma_a%20%3D%20%5CSigma_a%20-%20%5Cfrac%7B%5CSigma_aSS%5ET%5CSigma_a%7D%7B1%20&plus;%20S%5ET%5CSigma_aS%7D)

### Gaussian Bayes Critic

