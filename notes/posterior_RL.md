#WIP

# Using posterior sampling for exploration

These are my thoughts based on reading. Things here can be wrong, I'll try to point out where I might be assuming something that is not true with footnotes.

## Why can we not just use a posterior Q-value directly?

It seems reasonable that sampling the Q-value directly should account for our uncertainty. The issue with this comes from the Bellman equation.

$Q^*_\pi(s,a) = R(s,a) + \gamma\sum_{s'}P_{s'|s,a}\sum_{a'}\pi(a'|s')Q_\pi(s',a')$

The key is that the $Q$-value here is conditioned on a policy $\pi$. If we have a constant policy the problem is stationary<sup>[1](#1)</sup> and all the parametric uncertainty is contained in the posterior. <sup>[2](#2)</sup>

However, as soon as we start changing our policy, the expected total reward changes making the target non-stationary. Despite this, the function approximation methods used are derived for stationary targets. This means variance and posterior calculations will underestimate the uncertainty as they do not take into account the uncertainty in the policy which changes the Q-value. 

## Propagation of uncertainty

The way this is dealt with in the papers I have read is to try to propagate the uncertainty from future states. 


<a name="Footnote 1">1</a>: Even with a constant policy our target changes as we learn. However, from the way it is discussed in Sutton this is still considered stationary. I think this is because the return from the environment is stationary. How does this effect our uncertainty measure?

<a name="Footnote 2">2</a>: I'm still a bit unsure about this and this might be interesting to test just to include in the master. It should make sense that this is the case. Consider $Var(Q_{t}) = Var(r) + \gamma^2 Var(Q_{t+1})$. If the policy is constant the variance of $Q_{t}$ takes into account the future uncertainty. 