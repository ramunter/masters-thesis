# Documentation of Q-learning tests on a toy problem

## Corridor

This environment is a generalization of an environment called riverswim.

![Chain Env](./images/chain.png)

The corridor environment allows one to define:

* N: The environment consists of a N long corridor where the agent can move
  left or right.
* K: in a K number of states, the direction traveled when given is the
  opposite of the expected. I.e. action left will cause the agent to move right.
* p: Probability of success when moving right. 1-p probability of moving left
  instead.
        

## Experiment setup

The current setup gives 1 reward for reaching the final node to the right, and 0 otherwise. The agent has N steps to try to reach the goal. The experiment is run with K=0 and p=1 to allow a simple linear model and simple calculation of the optimal reward. 

The agent is then tested in the environment multiple times for an increasing value of N. The attempt is stopped when the agent reaches a running average of regret per episode that is lower than 1% of the optimal reward. When this happens we consider that the agent has "learned" the environment.


### Experiment pseudocode

<pre>
<b>for</b> agent <b>in</b> agents
  <b>for</b> N <b>in</b> cha<b>in</b>_lengths

    environment = corridor(N, 0, 1)

    <b>for</b> i <b>in</b> attempts
      steps_to_learn = q_learn(agent, environment)
    
    average_steps_to_learn = steps_to_learn/attempts
</pre>