# Reflection Week 12

## Overview of what has been done so far

### Using simple linear function Q-learning

* Implemented regular E greedy Q-learning.
* Implemented Gaussian Regression Q-learning for both known and unknown noise.
* Implemented UBE method without the uncertainty bellman equation but with sampled targets.
* Implemented a simple corridor environment with sparse rewards

Here the bayesian metods far outperform the other methods. However including unknown noise leads to unstable results. This means the method fails to learn the policy relatively often no matter the corridor length. If it does work, it learns it quickly. 

In an attempt to mimic large policy shifts I've tested flipping the action mapping(i.e left=right and vice versa). Most methods fail here, and E greedy can actually outperfor the gaussian regression.

### More complicated environments.

* Implemented the deep sea environment, but failed to get any good results using linear regression. I have not tested with neural networks.
* Added a bayesian DQN implentation similar to the one in [1]. Note that I sample coefficients every step.

The DQN is trained as usual, but use final layer encoding for each action as the input to a bayesian regression. This means that there is one bayesian regression model per action. Due to numerical issues, the covariance matrix ends up being non-symetric causing the tf multivariate sampling method to crash. Therefore I've had to only sample using the diag of the covariance matrix.

The bayesian regression is updated the same was as the DQN, i.e. using a batch sampled from the experience buffer. This method is extremely slow, running 17 fps on EPIC. Note that this is  on 512 final layer, which doesn't help the run time. This is to safely compare with DQN results. The results are pretty terrible. On cartpole it performs quite well to begin with, but learns slower than E greedy. It then completely fails again if you let it keep running. Overfitting I guess. This is the case for E greedy aswell, but for the bayesian method the performance drops to 20-40, while E greedy drops to a bit over 100. On pong in Atari it is also running worse than a regular DQN.

## Recent reading

I've spent some time this past week reading up on a few papers. This generally revolves around [1] and [2], along with some reading on relevant papers by Ian Osband. In addition I've been reading through the discussions around [1] from when it was rejected([3]). Their regret bound seems crazy good and is probably wrong, but there are a lot of interesting notes in the discussion, so I'd recommend reading through it.

Here are some of the ideas I think are interesting from these papers/discussions. Some are concepts we've discussed before:

1. [4] indicates that it is important to model the noise term aswell (according to [3]).
2. [2] indicates that the learning rate in tabular Q-learning is very important for the regret bound. I have an feeling that this can be encorperated in a kalman filter over the bayesian regression parameters, but need to think this through.
3. I know we use this as a simplifying assumption but [3] note that the Q-value is clearly not Normally distributed due to the max operation. Maybe there is a better distribution we could simplify to?


> [1] [Bayesian DQN Paper](https://arxiv.org/pdf/1802.04412.pdf)  
> [2] [Regret bounds in UCB Q-learning](https://arxiv.org/abs/1807.03765)  
> [3] [Bayesian DQN openreview](https://openreview.net/forum?id=B1e7hs05Km)  
> [4] [Bandit showdown](https://arxiv.org/pdf/1802.09127.pdf)


## Moving forward

Overall I want to get a DQN up and running that uses a similar setup to [1] but that encorporates the uncertainty propagation, the noise term and filters the parameters.

I think things that should be completed are:

* Get an overview over what is causing the failure of the BDQN. I think logging some parameter values and Q-values can give an indication of what is going on.
* Look into using the logic from [2] on a regret bound for Q-learning using PSRL.
* Add the deep sea env to dopamine.
* Kalman filter on the bayesian regression.
* Look into a more computational efficient setup and deal with the numerical problem with the covariance problem.