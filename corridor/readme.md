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

The agent is then tested in the environment multiple times for an increasing value of N. The attempt is stopped when the agent reaches a running average of regret per episode (optimal return - actual return) that is lower than 1% of the optimal reward. When this happens we consider that the agent has "learned" the environment.

Given the deterministic state transitions and N possible steps the optimal return is 1.


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

The agent is a vanilla Q-learning algorithm. The following pseudocode is essentialy a reproduction of the actual code to make the critic pseudocode clear-

<pre>
<b>Q-learner</b>

<b>for</b> episode <b>in</b> episodes
  state, critic = reset environment and critic
  action = critic.get_action(state)

  <b>while</b> not done:

    next_state, reward, done = take_step(action)

    next_action, next_q_value = critic.get_target_action_and_q_value(next_state)

    target = calculate_target(gamma, reward, next_q_value, done)

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


<b>Get Next Action and Q Value</b>

q_values = LinearRegressionModel(state, action | state) 
action = argmax q_values
Q_value = q_values[action]
</pre>

### Sample Target UBE Critic

Note that this implementation does not directly propagate the local uncertainty.


<pre>


<b>Get Action</b>

action = argmax Sample Q-value(state, action | state)


<b>Get Next Action and Q Value</b>
q_values = Sample Q-value(state, action | state)
next_action = argmax q_values
next_q_value = q_values[next_action]


<b>Sample Q-value</b>

mean_q = LinearRegressionModel(state, action)
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

<a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\textbf{Action&space;Variance}&space;\\\\&space;S^T\Sigma_aS&space;\\\\&space;\textbf{Update&space;}\Sigma_a&space;\\\\&space;\Sigma_a&space;=&space;\Sigma_a&space;-&space;\frac{\Sigma_aSS^T\Sigma_a}{1&space;&plus;&space;S^T\Sigma_aS}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\textbf{Action&space;Variance}&space;\\\\&space;S^T\Sigma_aS&space;\\\\&space;\textbf{Update&space;}\Sigma_a&space;\\\\&space;\Sigma_a&space;=&space;\Sigma_a&space;-&space;\frac{\Sigma_aSS^T\Sigma_a}{1&space;&plus;&space;S^T\Sigma_aS}" title="\\ \textbf{Action Variance} \\\\ S^T\Sigma_aS \\\\ \textbf{Update }\Sigma_a \\\\ \Sigma_a = \Sigma_a - \frac{\Sigma_aSS^T\Sigma_a}{1 + S^T\Sigma_aS}" /></a>

The intitial Sigma is set to an identity matrix of correct dimension.

### Gaussian Bayes Critic


<pre>


<b>Get Action</b>
coef = Sample N(mu, Sigma)
action = argmax BayesianRegressionModel(state, action, coef)


<b>Get Next Action and Q Value</b>
coef = Sample N(mu, Sigma)
q_values = BayesianRegressionModel(state, action, coef)
next_action = argmax q_values
next_q_value = q_values[next_action]

</pre>

The bayesian regression model is updated using the following update functions

<!-- 
$
\\
\textbf{Update Mean Parameters}
\\\\
\mu = (X^T X + \sigma_\varepsilon\Sigma^{-1})^{-1}(X^TY + \Sigma^{-1}\mu)
\\\\
\textbf{Update Covariance Matrix}
\\\\
\Sigma = (\sigma_\varepsilon^{-2}X^TX+\Sigma^{-1})^{-1}
$ -->

<a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\textbf{Update&space;Mean&space;Parameters}&space;\\\\&space;\mu&space;=&space;(X^T&space;X&space;&plus;&space;\sigma_\varepsilon\Sigma^{-1})^{-1}(X^TY&space;&plus;&space;\Sigma^{-1}\mu)&space;\\\\&space;\textbf{Update&space;Covariance&space;Matrix}&space;\\\\&space;\Sigma&space;=&space;(\sigma_\varepsilon^{-2}X^TX&plus;\Sigma^{-1})^{-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\textbf{Update&space;Mean&space;Parameters}&space;\\\\&space;\mu&space;=&space;(X^T&space;X&space;&plus;&space;\sigma_\varepsilon\Sigma^{-1})^{-1}(X^TY&space;&plus;&space;\Sigma^{-1}\mu)&space;\\\\&space;\textbf{Update&space;Covariance&space;Matrix}&space;\\\\&space;\Sigma&space;=&space;(\sigma_\varepsilon^{-2}X^TX&plus;\Sigma^{-1})^{-1}" title="\\ \textbf{Update Mean Parameters} \\\\ \mu = (X^T X + \sigma_\varepsilon\Sigma^{-1})^{-1}(X^TY + \Sigma^{-1}\mu) \\\\ \textbf{Update Covariance Matrix} \\\\ \Sigma = (\sigma_\varepsilon^{-2}X^TX+\Sigma^{-1})^{-1}" /></a>

### Deep Exploration Gaussian Bayes Critic

This algorithm is almost the same as the Gaussian Bayes Critic. The only difference is coefficients are sampled per episode rather than per step.
