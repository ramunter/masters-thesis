# Key Papers

This document will contain papers I feel will play an important role in guiding the master thesis.

## [Deep Bayesian Bandits Showdown](https://arxiv.org/pdf/1802.09127.pdf)

*Compares posterior sampling methods on different bandit problems. Shows generally bad results when using posterior sampling.*


### Abstract 

Recent advances in deep reinforcement learning have made significant strides in
performance on applications such as Go and Atari games. However, developing
practical methods to balance exploration and exploitation in complex domains
remains largely unsolved. Thompson Sampling and its extension to reinforcement
learning provide an elegant approach to exploration that only requires access
to posterior samples of the model. At the same time, advances in approximate
Bayesian methods have made posterior approximation for flexible neural network
models practical. Thus, it is attractive to consider approximate Bayesian neural
networks in a Thompson Sampling framework. To understand the impact of using
an approximate posterior on Thompson Sampling, we benchmark well-established
and recently developed methods for approximate posterior sampling combined
with Thompson Sampling over a series of contextual bandit problems. We found
that many approaches that have been successful in the supervised learning setting
underperformed in the sequential decision-making scenario. In particular, we
highlight the challenge of adapting slowly converging uncertainty estimates to the
online setting.

## [The Uncertainty Bellman Equation and Exploration](https://arxiv.org/pdf/1709.05380v3.pdf)

### Abstract 

We consider the exploration/exploitation problem in reinforcement learning. For exploitation, it is well known that the Bellman equation connects the value at any time-step to the expected value at subsequent time-steps. In this paper we  consider a similar uncertainty Bellman equation (UBE), which connects the uncertainty at any time-step to the expected uncertainties at subsequent time-steps, thereby  extending the potential exploratory benefit of a policy beyond individual time-steps. We prove that the unique fixed point of the UBE yields an upper bound on the variance of the posterior distribution of the Q-values induced by any policy. This bound can be much
tighter than traditional count-based bonuses that compound standard deviation rather than variance. Importantly, and unlike several existing approaches to optimism, this method scales naturally to large systems with complex generalization. Substituting our UBE-exploration strategy for -greedy improves DQN perform.



## [Successor Uncertainties: Exploration and Uncertainty in Temporal Difference Learning](https://arxiv.org/pdf/1810.06530.pdf)

*Propagates uncertainty but does not assume that $Q(s,a)$'s are independent, unlike UBE. Have not properly read the theory yet. The paper was rejected from ICLR 2019 for lacking novelty. Either way it contains some interesting discussions and references.* 

### Abstract

We consider the problem of balancing exploration and exploitation in sequential decision making problems. This trade-off naturally lends itself to probabilistic modelling. For a probabilistic approach to be effective, considering uncertainty about all immediate and long-term consequences of agent’s actions is vital. An estimate of such uncertainty can be leveraged to guide exploration even in situations where the agent needs to perform a potentially long sequence of actions before reaching an under-explored area of the environment. This observation was made by the authors of the Uncertainty Bellman Equation model (O’Donoghue et al., 2018), which explicitly considers full marginal uncertainty for each decision the agent faces. However, their model still considers a fully factorised posterior over the consequences of each action, meaning that dependencies vital for correlated long-term exploration are ignored. We go a step beyond and develop Successor Uncertainties, a probabilistic model for the state-action value function of a Markov Decision Process with a non-factorised covariance. We demonstrate how this leads to greatly improved performance on classic tabular exploration benchmarks and show strong performance of our method on a subset of Atari baselines. Overall, Successor Uncertainties provides a better probabilistic model for temporal difference learning at a similar computational cost to its predecessors.

## [Randomized Prior Functions for Deep Reinforcement Learning](https://arxiv.org/pdf/1806.03335.pdf)

*Ian Osband has written a whole bunch of papers on posterior sample based RL for his PhD. This paper is the product of that PhD. Seems like a cool idea but the results are slightly disappointing(Same as UBE). The paper gives a good overview of shortcomings of other methods. It claims that it does consider the 'multi-step' uncertainty but I'm not so sure. His argumentation is "Since each network k trains only on its own target value, BootDQN+prior propagates a temporally-consistent sample of Q-value". But each network is still trained on an everchanging policy, so how does that help?*

### Abstract

Dealing with uncertainty is essential for efficient reinforcement learning. There is a growing literature on uncertainty estimation for deep learning from fixed datasets, but many of the most popular approaches are poorly suited to sequential decision  problems. Other methods, such as bootstrap sampling, have no mechanism for uncertainty that does not come from the observed data. We highlight why this can be a crucial shortcoming and propose a simple remedy through addition of a randomized untrainable ‘prior’ network to each ensemble member. We prove that this approach is efficient with linear representations, provide simple illustrations of its efficacy with nonlinear representations and show that this approach scales to large-scale problems far better than previous attempts.