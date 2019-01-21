# WIP

# Using posterior sampling for exploration

These are my thoughts based on reading. Things here can be wrong, I'll try to point out where I might be assuming something that is not true with footnotes.

## Why can we not just use a posterior Q-value directly?

It seems reasonable that sampling the Q-value directly should account for our uncertainty. However, there is an issue with this that originates from the Bellman equation.

$Q^*(s,a|\pi) = R(s,a) + \gamma\sum_{s'}P_{s'|s,a}\sum_{a'}\pi(a'|s')Q_\pi(s',a'|\pi)$

The $Q$-value here is conditioned on a policy $\pi$. If we have a constant policy the problem is stationary<sup>[1](#1)</sup> and all the parametric uncertainty is contained in the posterior. 

However, as soon as we start changing our policy, the expected total reward changes making the target non-stationary. Despite this, the function approximation methods used are derived for stationary targets. This means the variance and posterior calculations will underestimate the uncertainty as they do not take into account the uncertainty in the policy of future Q-values.<sup>[2](#2)</sup>

To deal with this most papers attempt to "propagate" the variance from future Q-values into the current Q-value. This process means we cannot sample the posterior directly. Instead we must assume its distribution to be normal and sample based on our mean and propagated variance estimate.

## Can we remove the policy conditioning?

This problem is a result of the fact that we are assuming a constant policy. The understimation of variance can be viewed as residual variablity([Bayesian calibration of computer models](https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/1467-9868.00294)). Our regression model is missing a key parameter for estimation, the policy.

Ideally we want to find a Bellman equation for $Q^*(s,a,\pi)$ such that we can create a model that samples the posterior $Q^*(s,a,\pi)$. I believe that this might give the best possible exploration/explotation trade-off. However, I can't be the first to think of this so there are probably plently of obstacles stopping this.

Firstly the question is how do we define $\pi$?


---

<a name="Footnote 1">1</a>: Even with a constant policy our target changes as we learn. However, from the way it is discussed in Sutton this is still considered stationary. I think this is because the return from the environment is stationary. How does this effect our uncertainty measure?

<a name="Footnote 2">2</a>: Intuitively this isn't very clear and it would be interesting to test on a toy example just to include in the master. 


<a name="Footnote 3">3</a>: Based on this relatively unsource sentence in UBE - "By contrast, for any set of prior beliefs the optimal exploration policy can be computed directly by dynamic programming in the ayesian
belief space. "