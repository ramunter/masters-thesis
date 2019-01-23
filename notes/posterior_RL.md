# WIP

# Using posterior sampling for exploration

These are my thoughts based on reading. Things here can be wrong, I'll try to point out where I might be assuming something that is not true with footnotes.

## Why can we not just use a posterior Q-value directly?

It seems reasonable that sampling the Q-value directly should account for our uncertainty. However, there is an issue with this that originates from the Bellman equation.

$Q^*(s,a|\pi) = R(s,a) + \gamma\sum_{s'}P_{s'|s,a}\sum_{a'}\pi(a'|s')Q_\pi(s',a'|\pi)$

The $Q$-value here is conditioned on a policy $\pi$. If we have a constant policy the problem is stationary<sup>[1](#1)</sup> and all the parametric uncertainty is contained in the posterior. 

However, as soon as we start changing our policy, the expected total reward changes making the target non-stationary. Despite this, the function approximation methods used are derived for stationary targets. This means the variance and posterior calculations will underestimate the uncertainty as they do not take into account the uncertainty in the policy of future Q-values.<sup>[2](#2)</sup>

To deal with this most papers attempt to "propagate" the variance from future Q-values into the current Q-value. This process means we cannot sample the posterior directly. Instead we must assume its distribution to be normal and sample based on our mean and propagated variance estimate.

The way this is dealt with in the UBE is to create a Bellman like equation that relates the variance of the current Q-value to the variance of the next Q-value. This variance can then be learnt in the same way as Q-values are learnt. The downside to this is that the variance is an upper bound and one assumes that the posterior distribution is normal. In addition UBE assumes that Q-value uncertainties are uncorrelated, as seen by the covariance matrix when Thompson sampling.

(This part is based on a quick read, can be wrong) The successor uncertainty paper also does some form of propagation by including the Q value relations in the loss function through regularization terms. This incorporates the Q-value dependencies in contrast to UBE. However it once again assumes a normal posterior

## Directly sampling posterior
Ideally we want to find a Bellman equation for $Q^*(s,a,\pi)$ such that we can create a model that samples the posterior $Q^*(s,a,\pi)$. I believe that this might give the best possible exploration/explotation trade-off. However, I can't be the first to think of this so there are probably plently of obstacles stopping this.

Firstly the question is how do we define $\pi$?


---

<a name="Footnote 1">1</a>: Even with a constant policy our target changes as we learn. However, from the way it is discussed in Sutton this is still considered stationary. I think this is because the return from the environment is stationary. How does this effect our uncertainty measure?

There are many ways to approximate the posterior $Q$, however most of them do not perform any propagation and thus do not incorporate the policy uncertainty. The randomize prior function paper claims to do so, but I'm having a hard time seeing how it actually does this. However the results match UBE.

## Possible direction to move in

I think the ideal solution is to avoid having to propagate the uncertainty directly. To do this one needs to include the policy in the posterior. I think this indicates the need for an Actor-Critic method. Through these methods we can include a prior on the policy and try to use this to properly calculate the Q-value posterior.

## Bayesian Reinforcement Learning

Either way I need to get a better overview of bayesian reinforcement learning. I find the whole concept a bit confusing. The papers so far all consider a bayesian MDP such that we get priors on reward and transitions. Combining these gives a posterior sample of the Q-value. However the formula for a Q-value is conditioned on a policy.

---


<a name="Footnote 2">2</a>: Intuitively this isn't very clear and it would be interesting to test on a toy example just to include in the master. 


<a name="Footnote 3">3</a>: Based on this relatively unsource sentence in UBE - "By contrast, for any set of prior beliefs the optimal exploration policy can be computed directly by dynamic programming in the ayesian
belief space. "